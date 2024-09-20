#include "PhonemeClassifier.h"

#define DR_MP3_IMPLEMENTATION
#define DR_WAV_IMPLEMENTATION

#ifdef __GNUC__
#define TYPE1 long long unsigned int
#define TYPE2 long unsigned int
#else
#define TYPE1 size_t
#define TYPE2 size_t 
#endif

// Update this when adding/remove json things
#define CURRENT_VERSION 1
// Update this when modifying classifier parameters
#define CLASSIFIER_VERSION 4

#include <filesystem>
#include <dr_mp3.h>
#include <dr_wav.h>
#include <samplerate.h>
#include <mlpack/mlpack.hpp>
#include "ModelSerializer.h"

using namespace mlpack;
using namespace arma;

void PhonemeClassifier::Clip::loadMP3(int targetSampleRate) {
    // Read mp3 file from tsv
    //printf("Loading MP3: %s\n", tsvElements[TSVReader::Indices::PATH].c_str());

    drmp3_config cfg;
    drmp3_uint64 samples;

    std::string clipFullPath = clipPath + tsvElements[TSVReader::Indices::PATH];
    if (!std::filesystem::exists(clipFullPath)) {
        printf("%s does not exist\n", clipFullPath.c_str());
        return;
    }

    float* floatBuffer = drmp3_open_file_and_read_pcm_frames_f32(clipFullPath.c_str(), &cfg, &samples, NULL);

    if (floatBuffer == NULL) {
        printf("Failed to open file: %s\n", clipFullPath.c_str());
        return;
    }
    if (cfg.channels <= 0) {
        printf("%s has invalid channel count (%d)\n", clipFullPath.c_str(), cfg.channels);
        return;
    }
    if (cfg.sampleRate <= 0) {
        printf("%s has invalid sample rate (%d)\n", clipFullPath.c_str(), cfg.sampleRate);
        return;
    }

    float length = (double)samples / cfg.sampleRate;
    float newLength = length + 0.1f;
    if (newLength > allocatedLength) {
        // Not very efficient but should slow/stop after a certain point
        delete[] buffer;
        buffer = new float[(size_t)(targetSampleRate * newLength)];
        allocatedLength = newLength;
    }

    if (cfg.sampleRate == targetSampleRate) {
        memcpy(buffer, floatBuffer, sizeof(float) * samples);
        size = samples;
        sampleRate = cfg.sampleRate;
    } else if (cfg.sampleRate % targetSampleRate == 0 && cfg.sampleRate > targetSampleRate) {
        // Integer down sample rate conversion
        int factor = cfg.sampleRate / targetSampleRate;
        size = samples / factor;
        for (size_t i = 0; i < size; i++) {
            buffer[i] = floatBuffer[i * factor];
        }
        sampleRate = targetSampleRate;
    } else {
        SRC_DATA upsampleData = SRC_DATA();
        upsampleData.data_in = floatBuffer;
        upsampleData.input_frames = samples;
        double ratio = (double)targetSampleRate / cfg.sampleRate;
        long outSize = (long)(ratio * samples);
        upsampleData.src_ratio = ratio;
        upsampleData.data_out = buffer;
        upsampleData.output_frames = outSize;
        int error = src_simple(&upsampleData, SRC_SINC_BEST_QUALITY, cfg.channels);
        if (error) {
            std::cout << "Error while upsampling: " << src_strerror(error) << '\n';
        }
        sampleRate = targetSampleRate;
        size = outSize;
    }
    free(floatBuffer);

    // Normalize volume
    float max = 0;
    for (size_t i = 0; i < size; i++) {
        const float& val = buffer[i];
        if (val > max) {
            max = val;
        }
    }
    float targetMax = 1.0f;
    float factor = targetMax / max;
    for (size_t i = 0; i < size; i++) {
        buffer[i] *= factor;
    }

    loaded = true;
}

void PhonemeClassifier::initalize(const size_t& sr) {
    // Check if already initalized
    assert(!initalized);

    SAMPLE_RATE = sr;
    initalizePhonemeSet();
    outputSize = inversePhonemeSet.size();

    // Load JSON
    bool openedJson = json.open("classifier.json", CURRENT_VERSION);
    if (!openedJson) {
        json["classifier_version"] = CLASSIFIER_VERSION;
        json["input_features"] = (int)inputSize;
        json["output_features"] = (int)outputSize;
        json["training_seconds"] = 0.0;
        json.save();
    }
    int classifierVersion = json["classifier_version"].get_int();

    window = new float[FFT_FRAME_SAMPLES];
    for (int i = 0; i < FFT_FRAME_SAMPLES; i++) {
        //window[i] = 1.0; // None
        //window[i] = 0.5f * (1.0f - cos((6.2831853f * i) / FFT_FRAME_SAMPLES)); // Hann
        //window[i] = 0.5f * (1.0f - cos((6.2831853f * i) / FFT_FRAME_SAMPLES)) * pow(2.7182818f, (-5.0f * abs(FFT_FRAME_SAMPLES - 2.0f * i)) / FFT_FRAME_SAMPLES); // Hann - Poisson
        //window[i] = 0.355768f - 0.487396f * cosf((6.28318530f * i) / FFT_FRAME_SAMPLES) - 0.144232 * cosf((12.5663706f * i) / FFT_FRAME_SAMPLES) - 0.012604 * cosf((18.8495559f * i) / FFT_FRAME_SAMPLES); // Nuttall
        window[i] = 0.3635819 - 0.4891775 * cosf((6.28318530f * i) / FFT_FRAME_SAMPLES) - 0.1365995 * cosf((12.5663706f * i) / FFT_FRAME_SAMPLES) - 0.0106411 * cosf((18.8495559f * i) / FFT_FRAME_SAMPLES); // Blackman - Nuttall
    }

#pragma region Mel filterbank
    // Map fft to mel spectrum by index
    melTransform = new float* [MEL_BINS];
    melStart = new short[MEL_BINS];
    melEnd = new short[MEL_BINS];
    double melMax = 2595.0 * log10(1.0 + (sr / 700.0));
    double fftStep = (double)sr / FFT_REAL_SAMPLES;

    for (int melIdx = 0; melIdx < MEL_BINS; melIdx++) {
        melTransform[melIdx] = new float[FFT_REAL_SAMPLES];
        melStart[melIdx] = -1;
        melEnd[melIdx] = FFT_REAL_SAMPLES;
        double melFrequency = ((double)melIdx / MEL_BINS + (1.0 / MEL_BINS)) * melMax;

        for (int fftIdx = 0; fftIdx < FFT_REAL_SAMPLES; fftIdx++) {
            double frequency = (double)(fftIdx * sr) / FFT_REAL_SAMPLES;
            double fftFrequency = 2595.0 * log10(1.0 + (frequency / 700.0));

            double distance = abs(melFrequency - fftFrequency);
            distance /= fftStep * 2;
            double effectMultiplier = 1.0 - (distance * distance);

            if (effectMultiplier > 0 && melStart[melIdx] == -1) {
                melStart[melIdx] = fftIdx;
            } else if (effectMultiplier <= 0 && melStart[melIdx] != -1 && melEnd[melIdx] == FFT_REAL_SAMPLES) {
                melEnd[melIdx] = fftIdx;
            }

            effectMultiplier = std::max(0.0, effectMultiplier);
            melTransform[melIdx][fftIdx] = effectMultiplier;
        }
    }
    // Normalize
    for (int melIdx = 0; melIdx < MEL_BINS; melIdx++) {
        double sum = 0;
        for (int fftIdx = 0; fftIdx < FFT_REAL_SAMPLES; fftIdx++) {
            sum += melTransform[melIdx][fftIdx];
        }
        double factor = 1.0 / sum;
        for (int fftIdx = 0; fftIdx < FFT_REAL_SAMPLES; fftIdx++) {
            melTransform[melIdx][fftIdx] *= factor;
        }
    }
#pragma endregion

    fftwIn = (float*)fftw_malloc(sizeof(float) * FFT_FRAME_SAMPLES);
    fftwOut = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * FFT_REAL_SAMPLES);
    fftwPlan = fftwf_plan_dft_r2c_1d(FFT_FRAME_SAMPLES, fftwIn, fftwOut, FFTW_MEASURE | FFTW_DESTROY_INPUT);
    dctIn = (float*)fftw_malloc(sizeof(float) * MEL_BINS);
    dctOut = (float*)fftw_malloc(sizeof(float) * MEL_BINS);
    dctPlan = fftwf_plan_r2r_1d(MEL_BINS, dctIn, dctOut, FFTW_REDFT10, FFTW_MEASURE | FFTW_PRESERVE_INPUT);
    initalized = true;

    optimizer = ens::Adam(
        STEP_SIZE,  // Step size of the optimizer.
        0, // Batch size. Number of data points that are used in each iteration.
        0.9,        // Exponential decay rate for the first moment estimates.
        0.999, // Exponential decay rate for the weighted infinity norm estimates.
        1e-8,  // Value used to initialise the mean squared gradient parameter.
        0, // Max number of iterations.
        1e-8,           // Tolerance.
        true);

    bool loaded = false;
    std::vector<size_t> inputDimensions = { FRAME_SIZE, FFT_FRAMES, 1 };
    int savedInputSize = json["input_features"].get_int();
    int savedOutputSize = json["output_features"].get_int();
    bool metaMatch = (
        classifierVersion == CLASSIFIER_VERSION &&
        savedInputSize == inputSize &&
        savedOutputSize == outputSize);
    if (metaMatch && ModelSerializer::load(&network)) {
        std::cout << "Loaded model\n";
        loaded = true;
    }
    if (!loaded) {
        std::cout << "Model not loaded\n";
        json["training_seconds"] = 0.0;
        json["classifier_version"] = CLASSIFIER_VERSION;

        network.Add<LinearNoBias>(512);
        network.Add<LeakyReLU>();
        network.Add<LinearNoBias>(512);
        network.Add<LeakyReLU>();
        network.Add<LinearNoBias>(512);
        network.Add<LeakyReLU>();
        network.Add<LinearNoBias>(512);
        network.Add<LeakyReLU>();
        network.Add<Linear>(outputSize);
        network.Add<LogSoftMax>();
    }
    network.InputDimensions() = inputDimensions;
    optimizer.ResetPolicy() = false;

    ready = loaded;
    std::cout << "Classifier initalized\n\n";
}

void PhonemeClassifier::train(const std::string& path, const size_t& examples, const size_t& epochs, const double& stepSize) {
    optimizer.BatchSize() = 128;
    optimizer.StepSize() = stepSize;
    optimizer.MaxIterations() = epochs * examples * outputSize;

    TSVReader trainTSV;
    trainTSV.open(path + "/train.tsv");
    TSVReader testTSV;
    testTSV.open(path + "/test.tsv");
    const std::string clipPath = path + "/clips/";
    const std::string transcriptsPath = path + "/transcript/";

    std::vector<Phone> phones;
    
    double trainingSeconds = json["training_seconds"].get_real();
    size_t* phonemeTracker = new size_t[outputSize];
    for (size_t i = 0; i < outputSize; i++) {
        phonemeTracker[i] = 1;
    }

    std::thread trainThread;
    bool isTraining = false;
    size_t loops = 0;
    while (true) {
        std::printf("Starting loop %d\n", loops);
        loops++;
#pragma region Prepare to load data
        std::vector<mat> exampleData(outputSize);
        std::vector<mat> exampleLabel(outputSize);
        std::vector<size_t> exampleCount(outputSize);
        for (size_t i = 0; i < outputSize; i++) {
            exampleData[i] = mat(inputSize, examples);
            exampleLabel[i] = mat(1, examples);
            exampleLabel[i].fill(i);
            exampleCount[i] = 0;
        }

        std::vector<Frame> frames;
#pragma endregion
        size_t minExamples = 0;
        Clip clip = Clip();
        clip.initSampleRate(SAMPLE_RATE);
        size_t totalClips = 0;
        while ((minExamples < examples || isTraining) && minExamples < examples * MMAX_EXAMPLE_F) {
            loadNextClip(clipPath, trainTSV, clip, -1);

#pragma region Really shitty textgrid parser
            // Get transcription
            const std::string path = clip.tsvElements[TSVReader::Indices::PATH];
            const std::string transcriptionPath = transcriptsPath + clip.tsvElements[TSVReader::Indices::CLIENT_ID] + "/" + path.substr(0, path.length() - 4) + ".TextGrid";
            if (!std::filesystem::exists(transcriptionPath)) {
                //std::cout << "Missing transcription: " << transcriptionPath << std::endl;
                continue;
            }
            std::ifstream reader;
            reader.open(transcriptionPath);

            std::string line;
            while (reader.good()) {
                std::getline(reader, line);
                if (line == "        name = \"phones\" ") {
                    for (int i = 0; i < 2; i++) {
                        std::getline(reader, line);
                    }
                    break;
                }
            }
            std::getline(reader, line);
            std::string sizeString = line.substr(line.find_last_of(' ', line.length() - 2));
            int size = std::stoi(sizeString);
            phones = std::vector<Phone>(size);
            std::string interval, xmin, xmax, text;
            for (int i = 0; i < size; i++) {
                std::getline(reader, interval);
                std::getline(reader, xmin);
                std::getline(reader, xmax);
                std::getline(reader, text);

                xmin = xmin.substr(xmin.find_last_of(' ', xmin.length() - 2));
                xmax = xmax.substr(xmax.find_last_of(' ', xmax.length() - 2));
                size_t textStart = text.find_last_of('\"', text.length() - 3) + 1;
                text = text.substr(textStart, text.length() - textStart - 2);

                Phone p = Phone();
                p.min = std::stod(xmin);
                p.max = std::stod(xmax);
                p.minIdx = SAMPLE_RATE * p.min;
                p.maxIdx = SAMPLE_RATE * p.max;

                std::wstring wtext = utf8_to_utf16(text);

                const auto& pIterator = phonemeSet.find(customHasher(wtext));
                if (pIterator != phonemeSet.end()) {
                    p.phonetic = pIterator->second;
                } else {
                    std::cout << "unmapped phoneme: " << text << std::endl;
                    std::cout << "    at index " << i << " in " << transcriptionPath << std::endl;
                }
                phones[i] = p;
            }

            reader.close();
#pragma endregion

            // Check if the clip should be loaded
            bool shouldLoad = false;
            for (size_t i = 0; i < phones.size(); i++) {
                size_t binCount = exampleCount[phones[i].phonetic];
                if (binCount < examples * MMAX_EXAMPLE_F) {
                    shouldLoad = true;
                    break;
                }
            }
            if (!shouldLoad) {
                continue;
            }
            clip.loadMP3(SAMPLE_RATE);
            if (!clip.loaded) {
                continue;
            }
            totalClips++;
            
            // Process frames of the whole clip
            size_t fftStart = 0;
            size_t currentFrame = 0;
            trainingSeconds += (double)clip.size / clip.sampleRate;
            while ((size_t)fftStart + FFT_FRAME_SAMPLES < clip.size) {
                if (frames.size() <= currentFrame) {
                    frames.push_back(Frame());
                }
                Frame& frame = frames[currentFrame];
                Frame& prevFrame = (currentFrame > 0) ? frames[currentFrame - 1] : frame;
                frame.reset();

                PhonemeClassifier::processFrame(frame, clip.buffer, fftStart, clip.size, prevFrame);
#pragma region Phoneme overlap finder
                size_t maxOverlap = 0;
                size_t maxIdx = 0;
                for (int i = 0; i < size; i++) {
                    const Phone& p = phones[i];

                    if (p.maxIdx <= fftStart)
                        continue;
                    size_t fftEnd = fftStart + FFT_FRAME_SPACING;
                    if (p.minIdx >= fftEnd)
                        break;

                    size_t overlapA = p.maxIdx - fftStart;
                    size_t overlapB = fftEnd - p.minIdx;
                    size_t overlapC = FFT_FRAME_SPACING; // Window size
                    size_t overlapSize = std::min(std::min(overlapA, overlapB), overlapC);

                    if (overlapSize > maxOverlap) {
                        overlapSize = maxOverlap;
                        maxIdx = p.phonetic;
                    }
                }
                frame.phone = maxIdx;
#pragma endregion
                if (currentFrame >= FFT_FRAMES) {
                    const size_t& currentPhone = frames[currentFrame - CONTEXT_SIZE].phone;
                    size_t exampleIndex = exampleCount[currentPhone]++;

                    if (exampleIndex >= examples) {
                        // Random chance to not write anything
                        // Chance decreases as number of examples goes over max
                        double chance = (double)examples / exampleIndex;
                        int ran = rand();
                        double rnd = (double)ran / RAND_MAX;
                        if (rnd < chance) {
                            continue;
                        }
                        exampleIndex = ran % examples;
                    }

                    writeInput(frames, currentFrame, exampleData[currentPhone], exampleIndex);
                }

                fftStart += FFT_FRAME_SPACING;
                currentFrame++;
            }
            size_t minTemp = exampleCount[0];
            size_t minIdx = 0;
            for (size_t i = 1; i < outputSize; i++) {
                if (exampleCount[i] < minTemp) {
                    minTemp = exampleCount[i];
                    minIdx = i;
                }
            }
            minExamples = minTemp;
            if (!isTraining) {
                std::printf("%d clips; Min: %d of %s\r", (int)totalClips, (int)minExamples, inversePhonemeSet[minIdx].c_str());
                fflush(stdout);
            }
        }

        std::printf("Finished loading with minimum factor of %f\n", (double)minExamples / examples);

        if (trainThread.joinable()) {
            trainThread.join();
        }

        // Start training thread
        isTraining = true;
        bool copyDone = false;
        trainThread = std::thread([this, examples, &exampleData, &exampleLabel, &trainingSeconds, &isTraining, &copyDone]{
            size_t totalExamples = examples * outputSize;
            mat data = mat(inputSize, totalExamples);
            mat labels = mat(1, totalExamples);
            for (size_t i = 0; i < outputSize; i++) {
                const mat& copyFrom = exampleData[i];
                const mat& copyFromLabel = exampleLabel[i];
                size_t colOffset = examples * i;
                for (size_t c = 0; c < examples; c++) {
                    labels(0, colOffset + c) = copyFromLabel(0, c);
                    for (size_t r = 0; r < inputSize; r++) {
                        data(r, colOffset + c) = copyFrom(r, c);
                    }
                }
            }
            copyDone = true;

            // Calculate accuracy
#pragma region Calculate accuracy
            size_t correctCount = 0;
            size_t* correctPhonemes = new size_t[outputSize];
            size_t** confusionMatrix = new size_t*[outputSize];
            size_t* totalPhonemes = new size_t[outputSize];
            for (size_t i = 0; i < outputSize; i++) {
                totalPhonemes[i] = 0;
            }
            for (size_t i = 0; i < outputSize; i++) {
                correctPhonemes[i] = 0;
                totalPhonemes[i] = 0;
                confusionMatrix[i] = new size_t[outputSize];
                for (size_t j = 0; j < outputSize; j++) {
                    confusionMatrix[i][j] = 0;
                }
            }
            size_t testedExamples = 0;
            for (size_t i = 0; i < totalExamples; i++) {
                // 80% to skip / 20% to check
                double rnd = (double)rand() / RAND_MAX;
                if (rnd > 0.2) {
                    continue;
                }

                size_t result = classify(data.submat(span(0, inputSize - 1), span(i, i)));
                size_t label = labels(0, i);
                if (result == label) {
                    correctPhonemes[label]++;
                    correctCount++;
                }
                confusionMatrix[label][result]++;
                totalPhonemes[label]++;
                testedExamples++;
            }
            std::printf("Accuracy: %d out of %d (%.1f%%)\n", (int)correctCount, (int)testedExamples, ((double)correctCount / testedExamples) * 100);
            std::cout << "Confusion Matrix:\n";
            std::cout << "   ";
            for (size_t i = 0; i < outputSize; i++) {
                std::cout << std::setw(2) << inversePhonemeSet[i] << " ";
            }
            std::cout << std::endl;
            for (size_t i = 0; i < outputSize; i++) {
                std::cout << std::setw(2) << inversePhonemeSet[i] << " ";
                size_t total = 1;
                for (size_t j = 0; j < outputSize; j++) {
                    total += confusionMatrix[i][j];
                }
                for (size_t j = 0; j < outputSize; j++) {
                    double fraction = (double)confusionMatrix[i][j] / total;
                    int percent = fraction * 100;

                    const char* format = (i == j) ? "\033[36m%2d\033[0m " /* cyan */ :
                        (percent > 0) ? "\033[31m%2d\033[0m " /* red */ : "%2d ";

                    std::printf(format, percent);
                }
                std::cout << "\n";
            }

            delete[] totalPhonemes;
            delete[] correctPhonemes;
            for (size_t i = 0; i < outputSize; i++) {
                delete[] confusionMatrix[i];
            }
            delete[] confusionMatrix;
#pragma endregion

            network.Train(std::move(data),
                std::move(labels),
                optimizer,
                ens::PrintLoss(),
                ens::ProgressBar(),
                ens::EarlyStopAtMinLoss());

            std::cout << "Total hours of audio: " << trainingSeconds / (60 * 60) << "\n";
            json["training_seconds"] = trainingSeconds;
            json.save();

            ModelSerializer::save(&network);
            isTraining = false;
            });
        
        // Wait to finish copying data into new mat
        while (!copyDone) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
    }
}

size_t PhonemeClassifier::classify(const arma::mat& data) {
    mat results = mat();
    network.Predict(data, results);
    float max = results(0);
    size_t maxIdx = 0;
    for (size_t i = 0; i < outputSize; i++) {
        float val = results(i);
        if (val > max) {
            max = val;
            maxIdx = i;
        }
    }
    return maxIdx;
}

void PhonemeClassifier::processFrame(Frame& frame, const float* audio, const size_t& start, const size_t& totalSize, const Frame& prevFrame) {
    float max = 0.0;
    for (size_t i = 0; i < FFT_FRAME_SAMPLES; i++) {
        max = fmaxf(max, abs(audio[i]));
    }
    frame.volume = max * gain;
    for (size_t i = 0; i < FFT_FRAME_SAMPLES; i++) {
        size_t readLocation = (start + i) % totalSize;
        const float& value = audio[readLocation];
        fftwIn[i] = value * window[i] * gain;
    }

    // Get mel spectrum
    fftwf_execute(fftwPlan);
    float* fftAmplitudes = new float[FFT_REAL_SAMPLES];
    for (size_t i = 0; i < FFT_REAL_SAMPLES; i++) {
        fftwf_complex& complex = fftwOut[i];
        fftAmplitudes[i] = (complex[0] * complex[0] + complex[1] * complex[1]);
    }
    float* melFrequencies = new float[MEL_BINS];
    for (size_t i = 0; i < MEL_BINS; i++) {
        melFrequencies[i] = 0.0001f;
    }
    for (size_t melIdx = 0; melIdx < MEL_BINS; melIdx++) {
        for (size_t fftIdx = melStart[melIdx]; fftIdx < melEnd[melIdx]; fftIdx++) {
            const float& effect = melTransform[melIdx][fftIdx];
            melFrequencies[melIdx] += effect * fftAmplitudes[fftIdx];
        }
    }
    delete[] fftAmplitudes;

    // DCT of mel spectrum
    for (size_t i = 0; i < MEL_BINS; i++) {
        //frame.delta[i] = melAmplitude - prevFrame.real[i];
        //frame.real[i] = melAmplitude;
        dctIn[i] = log10f(melFrequencies[i]);
    }
    fftwf_execute(dctPlan);
    
    // Get the first FRAME_SIZE cepstrum coefficients
    float dctScale = 10.0f / (MEL_BINS * 2);
    for (size_t i = 0; i < FRAME_SIZE; i++) {
        float value = dctOut[i] * dctScale;
        //frame.real[i] = melFrequencies[i];
        frame.real[i] = value;
    }

    delete[] melFrequencies;
}

void PhonemeClassifier::writeInput(const std::vector<Frame>& frames, const size_t& lastWritten, mat& data, size_t col) {
    for (size_t f = 0; f < FFT_FRAMES; f++) {
        const Frame& readFrame = frames[(lastWritten + frames.size() - f) % frames.size()];
        size_t offset = f * FRAME_SIZE;
        for (size_t i = 0; i < FRAME_SIZE; i++) {
            data(offset + i, col) = readFrame.real[i];
            //data(offset + i * 2 + 1, col) = readFrame.delta[i];
        }
    }
}

void PhonemeClassifier::preprocessDataset(const std::string& path) {
    // Generate phoneme list
    TSVReader dictReader;
    dictReader.open(path + "/english_us_mfa.dict");
    std::vector<std::string> phonemeList = std::vector<std::string>();
    std::string* tabSeperated = dictReader.read_line_ordered();
    while (tabSeperated != NULL) {
        std::string& phonemes = tabSeperated[1];
        int start = 0;
        int end = 0;
        std::string p;
        while (end != -1) {
            end = phonemes.find_first_of(' ', start);
            p = phonemes.substr(start, end - start);
            start = end + 1;

            bool exists = false;
            for (size_t i = 0; i < phonemeList.size(); i++) {
                if (phonemeList[i] == p) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                phonemeList.push_back(p);
            }
        }
        tabSeperated = dictReader.read_line();
    }
    std::sort(phonemeList.begin(), phonemeList.end());
    std::fstream out = std::fstream("F:/Data/phonemes.txt", std::fstream::out);
    const char* newLine = "\n";
    for (size_t i = 0; i < phonemeList.size(); i++) {
        std::string& p = phonemeList[i];
        out.write(p.data(), p.size());
        out.write(newLine, 1);
    }
    out.close();

    // Generate phoneme alignments
    const std::vector<std::string> tables = {
        "/train.tsv",
        "/dev.tsv",
        "/test.tsv"
    };
    const std::string clipPath = path + "/clips/";
    const std::string corpusPath = "G:/corpus/";
    for (int i = 0; i < tables.size(); i++) {
        TSVReader tsv;
        tsv.open(path + tables[i].c_str());

        unsigned long globalCounter = 0;
        bool tsvGood = true;
        while (tsvGood) {
            auto start = std::chrono::system_clock::now();
            // Batches
            int counter = 0;
            Clip clip = Clip();
            clip.initSampleRate(SAMPLE_RATE);
            std::string* tabSeperated = tsv.read_line_ordered();
            while (tabSeperated != NULL && !(counter >= PREPROCESS_BATCH_SIZE && globalCounter % PREPROCESS_BATCH_SIZE == 0)) {
                std::cout << globalCounter << "\n";

                loadNextClip(clipPath, tabSeperated, clip, -1);

                globalCounter++;
                std::string transcriptPath = std::string(path) + "/transcript/" + clip.tsvElements[TSVReader::Indices::CLIENT_ID] + "/" + clip.tsvElements[TSVReader::Indices::PATH];
                transcriptPath = transcriptPath.substr(0, transcriptPath.length() - 4) + ".TextGrid";
                if (std::filesystem::exists(transcriptPath)) {
                    continue;
                }
                clip.loadMP3(16000);

                std::string speakerPath = corpusPath + clip.tsvElements[TSVReader::Indices::CLIENT_ID] + "/";
                if (!std::filesystem::is_directory(speakerPath)) {
                    std::filesystem::create_directory(speakerPath);
                }
                std::string originalPath = clip.tsvElements[TSVReader::Indices::PATH];
                std::string fileName = speakerPath + originalPath.substr(0, originalPath.length() - 4);

                // Convert audio to wav
                drwav wav;
                drwav_data_format format = drwav_data_format();
                format.container = drwav_container_riff;
                format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
                format.channels = 1;
                format.sampleRate = 16000;
                format.bitsPerSample = 32;
                drwav_init_file_write(&wav, (fileName + ".wav").c_str(), &format, NULL);
                drwav_uint64 framesWritten = drwav_write_pcm_frames(&wav, clip.size, clip.buffer);
                drwav_uninit(&wav);

                // Transcript
                std::ofstream fstream;
                fstream.open(fileName + ".txt");
                std::string sentence = clip.tsvElements[TSVReader::Indices::SENTENCE];
                fstream.write(sentence.c_str(), sentence.length());
                fstream.close();

                counter++;
            }
            // Run alignment
            // TODO: Change hardcoded paths
            system("conda activate aligner && \
              mfa align --clean \
                G:/corpus/ \
                F:/Data/cv-corpus-18.0-2024-06-14-en/en/english_us_mfa.dict \
                F:/Data/cv-corpus-18.0-2024-06-14-en/en/english_mfa.zip \
                F:/Data/cv-corpus-18.0-2024-06-14-en/en/transcript/");
            // Cleanup
            std::filesystem::directory_iterator iterator(corpusPath);
            for (const auto& directory : iterator) {
                std::filesystem::remove_all(directory);
            }
            auto diff = std::chrono::system_clock::now() - start;
            double duration = std::chrono::duration_cast<std::chrono::duration<double>>(diff).count();
            std::cout << "Iteration took " << duration << " seconds\n";
        }
    }
}

// https://stackoverflow.com/a/7154226
std::wstring PhonemeClassifier::utf8_to_utf16(const std::string& utf8) {
        std::vector<unsigned long> unicode;
        size_t i = 0;
        while (i < utf8.size())
        {
            unsigned long uni;
            size_t todo;
            bool error = false;
            unsigned char ch = utf8[i++];
            if (ch <= 0x7F)
            {
                uni = ch;
                todo = 0;
            } else if (ch <= 0xBF)
            {
                throw std::logic_error("not a UTF-8 string");
            } else if (ch <= 0xDF)
            {
                uni = ch & 0x1F;
                todo = 1;
            } else if (ch <= 0xEF)
            {
                uni = ch & 0x0F;
                todo = 2;
            } else if (ch <= 0xF7)
            {
                uni = ch & 0x07;
                todo = 3;
            } else
            {
                throw std::logic_error("not a UTF-8 string");
            }
            for (size_t j = 0; j < todo; ++j)
            {
                if (i == utf8.size())
                    throw std::logic_error("not a UTF-8 string");
                unsigned char ch = utf8[i++];
                if (ch < 0x80 || ch > 0xBF)
                    throw std::logic_error("not a UTF-8 string");
                uni <<= 6;
                uni += ch & 0x3F;
            }
            if (uni >= 0xD800 && uni <= 0xDFFF)
                throw std::logic_error("not a UTF-8 string");
            if (uni > 0x10FFFF)
                throw std::logic_error("not a UTF-8 string");
            unicode.push_back(uni);
        }
        std::wstring utf16;
        for (size_t i = 0; i < unicode.size(); ++i)
        {
            unsigned long uni = unicode[i];
            if (uni <= 0xFFFF)
            {
                utf16 += (wchar_t)uni;
            } else
            {
                uni -= 0x10000;
                utf16 += (wchar_t)((uni >> 10) + 0xD800);
                utf16 += (wchar_t)((uni & 0x3FF) + 0xDC00);
            }
        }
        return utf16;
    }

size_t PhonemeClassifier::customHasher(const std::wstring& str) {
    size_t v = 0;
    for (size_t i = 0; i < str.length(); i++) {
        v = (v << sizeof(wchar_t) * 8) ^ str[i];
    }
    return v;
}

void PhonemeClassifier::loadNextClip(const std::string& clipPath, std::string* tabSeperated, OUT Clip& clip, int sampleRate) {
    clip.clipPath = clipPath;
    clip.loaded = false;
    clip.tsvElements = tabSeperated;
    if (sampleRate > 0) {
        clip.loadMP3(sampleRate);
    }
}

void PhonemeClassifier::loadNextClip(const std::string& clipPath, TSVReader& tsv, OUT Clip& clip, int sampleRate) {
    std::string* elements = tsv.read_line();
    loadNextClip(clipPath, elements, clip, sampleRate);
}

void PhonemeClassifier::initalizePhonemeSet() {
#define REGISTER_PHONEME(t, p) \
    if (phonemeSet.find(PhonemeClassifier::customHasher(L##p)) != phonemeSet.end()) { \
        throw("Hash collision"); \
    } \
    phonemeSet[PhonemeClassifier::customHasher(L##p)] = _phonemeCounter++; \
    inversePhonemeSet.push_back(t);
#define REGISTER_ALIAS(p) \
    phonemeSet[PhonemeClassifier::customHasher(L##p)] = _phonemeCounter;

    size_t _phonemeCounter = 0;
    using namespace std::string_literals;
        REGISTER_PHONEME(""  , ""   )
        REGISTER_ALIAS("spn")
        REGISTER_ALIAS("ʔ")
        REGISTER_PHONEME("ai", "aj" )
        REGISTER_PHONEME("ow", "aw" )
        REGISTER_PHONEME("b" , "b"  )
        REGISTER_PHONEME("bj", "bʲ" )
        REGISTER_PHONEME("c" , "c"  )
        REGISTER_PHONEME("cg", "cʰ" )
        REGISTER_PHONEME("cw", "cʷ" )
        REGISTER_ALIAS("kʷ")
        REGISTER_PHONEME("d" , "d"  )
        REGISTER_ALIAS("ɾ")
        REGISTER_PHONEME("j" , "dʒ" )
        REGISTER_PHONEME("th", "dʲ" )
        REGISTER_PHONEME("du", "d̪" )
        REGISTER_PHONEME("e" , "ej" )
        REGISTER_PHONEME("f" , "f"  )
        REGISTER_PHONEME("fj", "fʲ" )
        REGISTER_PHONEME("h" , "h"  )
        REGISTER_PHONEME("i" , "i"  )
        REGISTER_PHONEME("i:", "iː" )
        REGISTER_PHONEME("j" , "j"  )
        REGISTER_PHONEME("k" , "k"  )
        REGISTER_PHONEME("kh", "kʰ" )
        REGISTER_PHONEME("l" , "l"  )
        REGISTER_PHONEME("m" , "m"  )
        REGISTER_ALIAS("m̩")
        REGISTER_ALIAS("ɱ")
        REGISTER_PHONEME("mj", "mʲ" )
        REGISTER_PHONEME("n" , "n"  )
        REGISTER_PHONEME("n.", "n̩" )
        REGISTER_PHONEME("o" , "ow" )
        REGISTER_PHONEME("p" , "p"  )
        REGISTER_ALIAS("pʷ")
        REGISTER_PHONEME("ph", "pʰ" )
        REGISTER_PHONEME("pj", "pʲ" )
        REGISTER_PHONEME("s" , "s"  )
        REGISTER_PHONEME("t" , "t"  )
        REGISTER_ALIAS("tʰ")
        REGISTER_ALIAS("tʲ")
        REGISTER_ALIAS("ɾʲ")
        REGISTER_PHONEME("tf", "tʃ" )
        REGISTER_PHONEME("tw", "tʷ" )
        REGISTER_PHONEME("th", "t̪" )
        REGISTER_PHONEME("v" , "v"  )
        REGISTER_PHONEME("vj", "vʲ" )
        REGISTER_PHONEME("w" , "w"  )
        REGISTER_PHONEME("z" , "z"  )
        REGISTER_PHONEME("ae", "æ"  )
        REGISTER_PHONEME("s" , "ç"  )
        REGISTER_PHONEME("th", "ð"  )
        REGISTER_PHONEME("ng", "ŋ"  )
        REGISTER_PHONEME("a" , "ɐ"  )
        REGISTER_PHONEME("a" , "ɑ"  )
        REGISTER_PHONEME("a:", "ɑː" )
        REGISTER_PHONEME("a" , "ɒ"  )
        REGISTER_PHONEME("a:", "ɒː" )
        REGISTER_PHONEME("o" , "ɔj" )
        REGISTER_PHONEME("uh", "ə"  )
        REGISTER_PHONEME("ah", "ɚ"  )
        REGISTER_PHONEME("eh", "ɛ"  )
        REGISTER_PHONEME("uh", "ɝ"  )
        REGISTER_PHONEME("g" , "ɟ"  )
        REGISTER_ALIAS("ɡ")
        REGISTER_PHONEME("gw", "ɟʷ" )
        REGISTER_ALIAS("ɡʷ")
        REGISTER_PHONEME("i" , "ɪ"  )
        REGISTER_PHONEME("l" , "ɫ"  )
        REGISTER_PHONEME("lw", "ɫ̩" )
        REGISTER_PHONEME("ny", "ɲ"  )
        REGISTER_ALIAS("ɾ̃")
        REGISTER_PHONEME("r" , "ɹ"  )
        REGISTER_PHONEME("sh", "ʃ"  )
        REGISTER_PHONEME("u" , "ʉ"  )
        REGISTER_PHONEME("u-", "ʉː" )
        REGISTER_PHONEME("oo", "ʊ"  )
        REGISTER_PHONEME("ll", "ʎ"  )
        REGISTER_PHONEME("z" , "ʒ"  )
        REGISTER_PHONEME("th", "θ"  )
#undef REGISTER_PHONEME
#undef ALIAS
}
