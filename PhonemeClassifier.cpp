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
#define CURRENT_VERSION 0
// Update this when modifying classifier parameters
#define CLASSIFIER_VERSION 0

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
    for (size_t i = 0; i < size; i++) {
        buffer[i] /= max;
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
    melTransform = new float* [FFT_REAL_SAMPLES];
    melStart = new short[FFT_REAL_SAMPLES];
    melEnd = new short[FFT_REAL_SAMPLES];
    double melMax = 2595.0 * log10(1.0 + (sr / 700.0));
    double fftStep = (double)sr / FFT_REAL_SAMPLES;
    for (int fftIdx = 0; fftIdx < FFT_REAL_SAMPLES; fftIdx++) {
        double frequency = (double)(fftIdx * sr) / FFT_REAL_SAMPLES;
        double melFrequency = 2595.0 * log10(1.0 + (frequency / 700.0));
        melTransform[fftIdx] = new float[FFT_REAL_SAMPLES];
        melStart[fftIdx] = -1;
        melEnd[fftIdx] = FFT_REAL_SAMPLES;
        for (int melIdx = 0; melIdx < FFT_REAL_SAMPLES; melIdx++) {
            double melBinFrequency = ((double)melIdx / FFT_REAL_SAMPLES) * melMax;
            double distance = abs(melFrequency - melBinFrequency);
            distance /= fftStep * 1.75;
            double effectMultiplier = 1.0 - (distance * distance);

            if (effectMultiplier > 0 && melStart[fftIdx] == -1) {
                melStart[fftIdx] = melIdx;
            } else if (effectMultiplier <= 0 && melStart[fftIdx] != -1 && melEnd[fftIdx] == FFT_REAL_SAMPLES) {
                melEnd[fftIdx] = melIdx;
            }

            effectMultiplier = std::max(0.0, effectMultiplier);
            melTransform[fftIdx][melIdx] = effectMultiplier;
        }
    }
    // Normalize
    for (int fftIdx = 0; fftIdx < FFT_REAL_SAMPLES; fftIdx++) {
        double sum = 0;
        for (int melIdx = 0; melIdx < FFT_REAL_SAMPLES; melIdx++) {
            sum += melTransform[fftIdx][melIdx];
        }
        for (int melIdx = 0; melIdx < FFT_REAL_SAMPLES; melIdx++) {
            melTransform[fftIdx][melIdx] /= sum;
        }
    }
#pragma endregion

    fftwIn = (float*)fftw_malloc(sizeof(float) * FFT_FRAME_SAMPLES);
    fftwOut = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * FFT_REAL_SAMPLES);
    fftwPlan = fftwf_plan_dft_r2c_1d(FFT_FRAME_SAMPLES, fftwIn, fftwOut, FFTW_MEASURE | FFTW_DESTROY_INPUT);
    dctIn = (float*)fftw_malloc(sizeof(float) * FFT_REAL_SAMPLES);
    dctOut = (float*)fftw_malloc(sizeof(float) * FFT_REAL_SAMPLES);
    dctPlan = fftwf_plan_r2r_1d(FFT_REAL_SAMPLES, dctIn, dctOut, FFTW_REDFT10, FFTW_MEASURE | FFTW_PRESERVE_INPUT);
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
    if (classifierVersion == CLASSIFIER_VERSION && ModelSerializer::load(&network)) {
        std::cout << "Loaded model\n";
        loaded = true;
    }
    if (!loaded) {
        std::cout << "Model not loaded\n";
        json["training_seconds"] = 0.0;

        network.Add<LinearType<mat, L2Regularizer>>(256, L2Regularizer(0.0001));
        network.Add<LeakyReLU>();
        network.Add<LinearType<mat, L2Regularizer>>(256, L2Regularizer(0.0001));
        network.Add<LeakyReLU>();
        network.Add<LinearType<mat, L2Regularizer>>(256, L2Regularizer(0.0001));
        network.Add<LeakyReLU>();
        network.Add<LinearType<mat, L2Regularizer>>(outputSize, L2Regularizer(0.0001));
        network.Add<LogSoftMax>();
    }
    network.InputDimensions() = inputDimensions;
    optimizer.ResetPolicy() = false;

    ready = loaded;
    std::cout << "Classifier initalized\n\n";
}

void PhonemeClassifier::train(const std::string& path, const size_t& batchSize, const size_t& epochs, const double& stepSize) {
    optimizer.BatchSize() = 16;
    optimizer.ResetPolicy() = false;
    optimizer.StepSize() = stepSize;

    TSVReader trainTSV;
    trainTSV.open(path + "/train.tsv");
    TSVReader testTSV;
    testTSV.open(path + "/test.tsv");
    const std::string clipPath = path + "/clips/";
    const std::string transcriptsPath = path + "/transcript/";

    std::vector<Clip> clips = std::vector<Clip>(batchSize);
    for (size_t i = 0; i < clips.size(); i++) {
        clips[i].initSampleRate(SAMPLE_RATE);
    }
    std::vector<Phone> phones;
    
    double trainingSeconds = json["training_seconds"].get_real();
    size_t* phonemeTracker = new size_t[outputSize];
    for (size_t i = 0; i < outputSize; i++) {
        phonemeTracker[i] = 1;
    }

    std::thread trainThread;
    bool isTraining = false;
    while (true) {
#pragma region Prepare to load clips
        size_t clipCount = 0;
        size_t testClips = 0;
        for (size_t i = 0; i < batchSize; i++) {
            bool test = clipCount % 10 == 0;
            clips[i].isTest = test;
            TSVReader& targetReader = (test) ? testTSV : trainTSV;
            loadNextClip(clipPath, targetReader, clips[i], -1);
            if (test) {
                testClips++;
            }
            clipCount++;
        }

        size_t clipsLoaded = 0;
#pragma omp parallel for
        for (size_t i = 0; i < clipCount; i++) {
            clips[i].loadMP3(SAMPLE_RATE);
#pragma omp critical
            clipsLoaded++;
            if (!isTraining) {
                std::cout << clipsLoaded << "\r" << std::flush;
            }
        }

        size_t totalTrainFrames = 0;
        size_t totalTestFrames = 0;
        size_t actualLoadedClips = 0;
        for (size_t i = 0; i < clipCount; i++) {
            if (clips[i].loaded) {
                size_t clipFrames = (clips[i].size + FFT_FRAME_SAMPLES) / FFT_FRAME_SPACING;

                if (clips[i].isTest) {
                    totalTestFrames += clipFrames;
                } else {
                    totalTrainFrames += clipFrames;
                }
                actualLoadedClips++;
            }
        }

        std::cout << actualLoadedClips << " clips loaded out of " << clipCount << " total\n";
        std::cout << testClips << " test clips\n";

        if (trainThread.joinable()) {
            trainThread.join();
            std::cout << "Finished\n";
        }

        mat train = mat(inputSize, totalTrainFrames);
        mat trainLabels = mat(1, totalTrainFrames);
        mat test = mat(inputSize, totalTestFrames);
        mat testLabels = mat(1, totalTestFrames);
        train.fill(0.0);
        test.fill(0.0);

        size_t trainFrame = 0;
        size_t testFrame = 0;
        std::vector<Frame> frames;
        size_t* totalPhonemes = new size_t[outputSize];
        for (size_t i = 0; i < outputSize; i++) {
            totalPhonemes[i] = 0;
        }
#pragma endregion
        for (size_t c = 0; c < clipCount; c++) {
            Clip& currentClip = clips[c];
            if (!currentClip.loaded) {
                continue;
            }

#pragma region Really shitty textgrid parser
            // Get transcription
            const std::string path = currentClip.tsvElements[TSVReader::Indices::PATH];
            const std::string transcriptionPath = transcriptsPath + currentClip.tsvElements[TSVReader::Indices::CLIENT_ID] + "/" + path.substr(0, path.length() - 4) + ".TextGrid";
            if (!std::filesystem::exists(transcriptionPath)) {
                std::cout << "Missing transcription: " << transcriptionPath << std::endl;
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

            // Process frames of the whole clip
            size_t fftStart = 0;
            size_t currentFrame = 0;
            trainingSeconds += (double)currentClip.size / currentClip.sampleRate;

            while ((size_t)fftStart + FFT_FRAME_SAMPLES < currentClip.size) {
                if (frames.size() <= currentFrame) {
                    frames.push_back(Frame());
                }
                Frame& frame = frames[currentFrame];
                Frame& prevFrame = (currentFrame > 0) ? frames[currentFrame - 1] : frame;
                frame.reset();

                PhonemeClassifier::processFrame(frame, currentClip.buffer, fftStart, currentClip.size, prevFrame);
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
                totalPhonemes[maxIdx]++;
#pragma endregion
                if (currentFrame >= FFT_FRAMES) {
                    bool isTest = currentClip.isTest;
                    if (isTest) {
                        writeInput(frames, currentFrame, test, testFrame);
                        testLabels(0, testFrame) = maxIdx;
                        testFrame++;
                    } else {
                        writeInput(frames, currentFrame, train, trainFrame);
                        trainLabels(0, trainFrame) = maxIdx;
                        trainFrame++;
                    }
                }

                fftStart += FFT_FRAME_SPACING;
                currentFrame++;
            }
            if (!isTraining) {
                std::cout << c << "\r" << std::flush;
            }
        }
        train = train.submat(0, 0, inputSize - 1, trainFrame - 1);
        trainLabels = trainLabels.submat(0, 0, 0, trainFrame - 1);
        test = test.submat(0, 0, inputSize - 1, testFrame - 1);
        testLabels = testLabels.submat(0, 0, 0, testFrame - 1);

        std::cout << "Starting under sampling\n";
#pragma region Under sampling
        size_t droppedFrames = 0;
        uvec dropIndices = uvec();
        size_t dropIdx = 0;
        const double targetFraction = 1.0 / outputSize;
        const double tolerance = targetFraction * 0.01;
        const double samplingFactor = 0.75;
        size_t trackerTotal = 0;
        for (size_t i = 0; i < outputSize; i++) {
            trackerTotal += phonemeTracker[i];
        }
        int64_t targetPhonemeCount = targetFraction * trackerTotal;
        std::cout << "Target: " << trainFrame * targetFraction << std::endl;
        std::vector<std::vector<TYPE1>> phonemeCols = std::vector<std::vector<TYPE1>>(outputSize);
        for (size_t i = 0; i < trainFrame; i++) {
            const size_t& label = trainLabels(0, i);
            phonemeCols[label].push_back(i);
        }
        for (size_t i = 0; i < outputSize; i++) {
            size_t totalCount = phonemeCols[i].size();
            std::cout << inversePhonemeSet[i] << ": " << totalCount <<
                " / " << phonemeTracker[i];

            // How much higher it was than target
            // Negative if higher, positive if lower
            int64_t historicalOffset = targetPhonemeCount - phonemeTracker[i];
            if (historicalOffset >= -tolerance * trackerTotal) {
                std::cout << std::endl;
                phonemeTracker[i] += totalCount;
                continue;
            }

            if (historicalOffset < -tolerance * trackerTotal) {
                // Under sampling
                int64_t dropAmount = historicalOffset * -samplingFactor;
                dropAmount = std::min(dropAmount, (int64_t)(totalCount * samplingFactor));
                std::cout << " (Dropping " << dropAmount << " frames)" << std::endl;
                dropIndices.resize(dropIndices.size() + dropAmount);
                droppedFrames += dropAmount;
                phonemeTracker[i] += totalCount - dropAmount;
                for (int j = totalCount - 1; j > 0; j--) {
                    double random = (double)rand() / RAND_MAX;
                    if (random < ((double)dropAmount / j)) {
                        dropIndices(dropIdx) = phonemeCols[i][j];
                        dropIdx++;
                        dropAmount--;
                    }
                }
            }
        }
        // Apply changes to training data
        train.shed_cols(dropIndices);
        trainLabels.shed_cols(dropIndices);
#pragma endregion

        std::cout << trainFrame << " starting train frames\n";
        trainFrame -= droppedFrames;
        std::cout << droppedFrames << " dropped frames\n";
        std::cout << trainFrame << " effective train frames\n";
        std::cout << testFrame << " test frames\n";
        optimizer.MaxIterations() = epochs * trainFrame;

        // Calculate accuracy
#pragma region Calculate accuracy
        size_t correctCount = 0;
        size_t* correctPhonemes = new size_t[outputSize];
        size_t** confusionMatrix = new size_t * [outputSize];
        for (size_t i = 0; i < outputSize; i++) {
            correctPhonemes[i] = 0;
            totalPhonemes[i] = 0;
            confusionMatrix[i] = new size_t[outputSize];
            for (size_t j = 0; j < outputSize; j++) {
                confusionMatrix[i][j] = 0;
            }
        }
        for (size_t i = 0; i < testFrame; i++) {
            size_t result = classify(test.submat(span(0, inputSize - 1), span(i, i)));
            size_t label = testLabels(0, i);
            if (result == label) {
                correctPhonemes[label]++;
                correctCount++;
            }
            confusionMatrix[label][result]++;
            totalPhonemes[label]++;
        }
        std::printf("Accuracy: %d out of %d (%.1f%%)\n", (int)correctCount, (int)testFrame, ((double)correctCount / testFrame) * 100);
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
                    (confusionMatrix[i][j] > confusionMatrix[i][i]) ? "\033[31m%2d\033[0m " /* red */ : "%2d ";

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

        isTraining = true;
        trainThread = std::thread([this, train, trainLabels, &trainingSeconds, &isTraining]{
            network.Train(std::move(train),
                std::move(trainLabels),
                optimizer,
                ens::PrintLoss(),
                ens::ProgressBar(),
                ens::EarlyStopAtMinLoss());

            std::cout << "Total hours trained: " << trainingSeconds / (60 * 60) << "\n";
            json["training_seconds"] = trainingSeconds;
            json.save();

            ModelSerializer::save(&network);
            isTraining = false;
            });
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
        fftwIn[i] = value * window[i];
    }

    // Get mel spectrum
    fftwf_execute(fftwPlan);
    float* melFrequencies = new float[FFT_REAL_SAMPLES];
    for (size_t i = 0; i < FFT_REAL_SAMPLES; i++) {
        melFrequencies[i] = 0.0001f;
    }
    for (size_t i = 0; i < FFT_REAL_SAMPLES; i++) {
        fftwf_complex& complex = fftwOut[i];
        float amplitude = (complex[0] * complex[0] + complex[1] * complex[1]) * gain;
        for (size_t melIdx = melStart[i]; melIdx < melEnd[i]; melIdx++) {
            const float& effect = melTransform[i][melIdx];
            melFrequencies[melIdx] += effect * (amplitude);
        }
    }

    // DCT of mel spectrum
    for (size_t i = 0; i < FFT_REAL_SAMPLES; i++) {
        //frame.delta[i] = melAmplitude - prevFrame.real[i];
        //frame.real[i] = melAmplitude;
        dctIn[i] = log10f(melFrequencies[i]);
    }
    fftwf_execute(dctPlan);
    
    // Get the first FRAME_SIZE cepstrum coefficients
    float dctScale = 10.0f / (FFT_REAL_SAMPLES * 2);
    for (size_t i = 0; i < FRAME_SIZE; i++) {
        float value = dctOut[i] * dctScale;
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
    phonemeSet[PhonemeClassifier::customHasher(L##p)] = _phonemeCounter++; \
    inversePhonemeSet.push_back(t);
#define REGISTER_ALIAS(p) \
    phonemeSet[PhonemeClassifier::customHasher(L##p)] = _phonemeCounter;

    size_t _phonemeCounter = 0;
    using namespace std::string_literals;
        REGISTER_PHONEME(""  , ""   )
        REGISTER_PHONEME("=" , "spn")
        REGISTER_PHONEME("ai", "aj" )
        REGISTER_PHONEME("ow", "aw" )
        REGISTER_PHONEME("b" , "b"  )
        REGISTER_ALIAS("bʲ" )
        REGISTER_PHONEME("c" , "c"  )
        REGISTER_ALIAS("cʰ" )
        REGISTER_ALIAS("cʷ" )
        REGISTER_PHONEME("d" , "d"  )
        REGISTER_PHONEME("j" , "dʒ" )
        REGISTER_PHONEME("th", "dʲ" )
        REGISTER_PHONEME("du", "d̪" )
        REGISTER_PHONEME("e" , "ej" )
        REGISTER_PHONEME("f" , "f"  )
        REGISTER_ALIAS("fʲ" )
        REGISTER_PHONEME("h" , "h"  )
        REGISTER_PHONEME("i" , "i"  )
        REGISTER_ALIAS("iː" )
        REGISTER_PHONEME("j" , "j"  )
        REGISTER_PHONEME("k" , "k"  )
        REGISTER_ALIAS("kʰ" )
        REGISTER_ALIAS("kʷ" )
        REGISTER_PHONEME("l" , "l"  )
        REGISTER_PHONEME("m" , "m"  )
        REGISTER_ALIAS("mʲ" )
        REGISTER_ALIAS("m̩" )
        REGISTER_PHONEME("n" , "n"  )
        REGISTER_PHONEME("n" , "n̩" )
        REGISTER_PHONEME("o" , "ow" )
        REGISTER_PHONEME("p" , "p"  )
        REGISTER_ALIAS("pʰ" )
        REGISTER_ALIAS("pʲ" )
        REGISTER_ALIAS("pʷ" )
        REGISTER_PHONEME("s" , "s"  )
        REGISTER_PHONEME("t" , "t"  )
        REGISTER_ALIAS("tʃ" )
        REGISTER_ALIAS("tʰ" )
        REGISTER_ALIAS("tʲ" )
        REGISTER_ALIAS("tʷ" )
        REGISTER_PHONEME("th", "t̪" )
        REGISTER_PHONEME("v" , "v"  )
        REGISTER_ALIAS("vʲ" )
        REGISTER_PHONEME("w" , "w"  )
        REGISTER_PHONEME("z" , "z"  )
        REGISTER_PHONEME("ae", "æ"  )
        REGISTER_PHONEME("s" , "ç"  )
        REGISTER_PHONEME("th", "ð"  )
        REGISTER_PHONEME("ng", "ŋ"  )
        REGISTER_PHONEME("a" , "ɐ"  )
        REGISTER_PHONEME("a" , "ɑ"  )
        REGISTER_ALIAS("ɑː" )
        REGISTER_PHONEME("a" , "ɒ"  )
        REGISTER_ALIAS("ɒː" )
        REGISTER_PHONEME("o" , "ɔj" )
        REGISTER_PHONEME("uh", "ə"  )
        REGISTER_PHONEME("ah", "ɚ"  )
        REGISTER_PHONEME("eh", "ɛ"  )
        REGISTER_PHONEME("uh", "ɝ"  )
        REGISTER_PHONEME("g" , "ɟ"  )
        REGISTER_ALIAS("ɟʷ" )
        REGISTER_PHONEME("g" , "ɡ"  )
        REGISTER_ALIAS("ɡʷ" )
        REGISTER_PHONEME("i" , "ɪ"  )
        REGISTER_PHONEME("l" , "ɫ"  )
        REGISTER_ALIAS("ɫ̩" )
        REGISTER_PHONEME("mf", "ɱ"  )
        REGISTER_PHONEME("ny", "ɲ"  )
        REGISTER_PHONEME("r" , "ɹ"  )
        REGISTER_PHONEME("tt", "ɾ"  )
        REGISTER_ALIAS("ɾʲ" )
        REGISTER_ALIAS("ɾ̃"  )
        REGISTER_PHONEME("sh", "ʃ"  )
        REGISTER_PHONEME("u" , "ʉ"  )
        REGISTER_ALIAS("ʉː" )
        REGISTER_PHONEME("oo", "ʊ"  )
        REGISTER_PHONEME("ll", "ʎ"  )
        REGISTER_PHONEME("z" , "ʒ"  )
        REGISTER_PHONEME("/" , "ʔ"  )
        REGISTER_PHONEME("th", "θ"  )
#undef REGISTER_PHONEME
}
