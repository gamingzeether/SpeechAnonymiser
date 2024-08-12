#include "PhonemeClassifier.h"

#define DR_MP3_IMPLEMENTATION
#define DR_WAV_IMPLEMENTATION

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
#pragma omp critical
    std::cout << "Loading MP3: " << tsvElements[TSVReader::Indices::PATH] << '\n';

    drmp3_config cfg;
    drmp3_uint64 samples;

    std::string clipFullPath = clipPath + tsvElements[TSVReader::Indices::PATH];
    if (!std::filesystem::exists(clipFullPath)) {
#pragma omp critical
        std::cout << clipFullPath << " does not exist\n";
        return;
    }

    float* floatBuffer = drmp3_open_file_and_read_pcm_frames_f32(clipFullPath.c_str(), &cfg, &samples, NULL);

    if (cfg.channels <= 0) {
#pragma omp critical
        std::cout << clipFullPath << " has invalid channel count (" << cfg.channels << ")\n";
        return;
    }
    if (cfg.sampleRate <= 0) {
#pragma omp critical
        std::cout << clipFullPath << " has invalid sample rate (" << cfg.sampleRate << ")\n";
        return;
    }

    if (cfg.sampleRate == targetSampleRate) {
        memcpy(buffer, floatBuffer, sizeof(float) * samples);
        size = samples;
        sampleRate = cfg.sampleRate;
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
    loaded = true;
}

void PhonemeClassifier::initalize(const size_t& sr, bool load) {
    SAMPLE_RATE = sr;
    initalizePhonemeSet();
    outputSize = phonemeSet.size();

    window = new float[FFT_FRAME_SAMPLES];
    for (int i = 0; i < FFT_FRAME_SAMPLES; i++) {
        //window[i] = 1.0; // None
        //window[i] = 0.5f * (1.0f - cos((6.2831853f * i) / FFT_FRAME_SAMPLES)); // Hann
        //window[i] = 0.5f * (1.0f - cos((6.2831853f * i) / FFT_FRAME_SAMPLES)) * pow(2.7182818f, (-5.0f * abs(FFT_FRAME_SAMPLES - 2.0f * i)) / FFT_FRAME_SAMPLES); // Hann - Poisson
        //window[i] = 0.355768f - 0.487396f * cosf((6.28318530f * i) / FFT_FRAME_SAMPLES) - 0.144232 * cosf((12.5663706f * i) / FFT_FRAME_SAMPLES) - 0.012604 * cosf((18.8495559f * i) / FFT_FRAME_SAMPLES); // Nuttall
        window[i] = 0.3635819 - 0.4891775 * cosf((6.28318530f * i) / FFT_FRAME_SAMPLES) - 0.1365995 * cosf((12.5663706f * i) / FFT_FRAME_SAMPLES) - 0.0106411 * cosf((18.8495559f * i) / FFT_FRAME_SAMPLES); // Blackman - Nuttall
    }

    fftwIn = (float*)fftw_malloc(sizeof(float) * FFT_FRAME_SAMPLES);
    fftwOut = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * FFT_REAL_SAMPLES);
    fftwPlan = fftwf_plan_dft_r2c_1d(FFT_FRAME_SAMPLES, fftwIn, fftwOut, FFTW_MEASURE);
    initalized = true;

    if (load) {
        optimizer = ens::Adam(
            STEP_SIZE,  // Step size of the optimizer.
            0, // Batch size. Number of data points that are used in each iteration.
            0.9,        // Exponential decay rate for the first moment estimates.
            0.999, // Exponential decay rate for the weighted infinity norm estimates.
            1e-8,  // Value used to initialise the mean squared gradient parameter.
            0, // Max number of iterations.
            1e-8,           // Tolerance.
            true);

        bool loaded = ModelSerializer::load((void*)&network);
        if (loaded) {
            std::cout << "Loaded model\n";
        } else {
            std::cout << "Model could not be loaded\n";

            network.Add<Linear>(1024);
            network.Add<LeakyReLU>();
            network.Add<Linear>(512);
            network.Add<LeakyReLU>();
            network.Add<Linear>(outputSize);
            network.Add<LogSoftMax>();
        }

        ready = loaded;
    }
}

void PhonemeClassifier::train(const std::string& path, const size_t& batchSize) {
    optimizer.BatchSize() = batchSize;

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

    while (trainTSV.good()) {
        size_t clipCount = 0;
        size_t testClips = 0;
        for (size_t i = 0; i < batchSize && trainTSV.good(); i++) {
            double random = (double)rand() / RAND_MAX;
            bool test = random < 0.1;
            clips[i].isTest = test;
            TSVReader& targetReader = (test) ? testTSV : trainTSV;
            loadNextClip(clipPath, targetReader, clips[i], -1);
            if (test) {
                testClips++;
            }
            clipCount++;
        }

#pragma omp parallel for
        for (size_t i = 0; i < clipCount; i++) {
            clips[i].loadMP3(SAMPLE_RATE);
        }

        size_t totalTrainFrames = 0;
        size_t totalTestFrames = 0;
        size_t actualLoadedClips = 0;
        size_t skipFrames = FFT_FRAMES / 2;
        for (size_t i = 0; i < clipCount; i++) {
            if (clips[i].loaded) {
                size_t clipFrames = (clips[i].size + FFT_FRAME_SAMPLES) / FFT_FRAME_SPACING;
                clipFrames = clipFrames / (FFT_FRAMES / skipFrames);

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
        std::cout << totalTrainFrames << " train frames\n";
        std::cout << totalTestFrames << " test frames\n";

        mat train = mat(inputSize, totalTrainFrames);
        mat trainLabels = mat(1, totalTrainFrames);
        mat test = mat(inputSize, totalTestFrames);
        mat testLabels = mat(1, totalTestFrames);
        optimizer.MaxIterations() = EPOCHS * totalTrainFrames;
        train.fill(0.0);
        test.fill(0.0);

        size_t trainFrame = 0;
        std::vector<Frame> frames;
#pragma omp parallel for
        for (intptr_t c = 0; c < clipCount; c++) {
            Clip& currentClip = clips[c];
            if (!currentClip.loaded) {
                continue;
            }

#pragma region Really shitty textgrid parser
            // Get transcription
            const std::string path = currentClip.tsvElements[TSVReader::Indices::PATH];
            const std::string transcriptionPath = transcriptsPath + currentClip.tsvElements[TSVReader::Indices::CLIENT_ID] + "/" + path.substr(0, path.length() - 4) + ".TextGrid";
            if (!std::filesystem::exists(transcriptionPath)) {
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
            size_t trainFrame = 0;
            size_t testFrame = 0;
            size_t sinceLastFrame = 0;

            while ((size_t)fftStart + FFT_FRAME_SAMPLES < currentClip.size) {
                if (frames.size() <= currentFrame) {
                    frames.push_back(Frame());
                }
                Frame& frame = frames[currentFrame];
                frame.reset();

                PhonemeClassifier::processFrame(frame, currentClip.buffer, fftStart, currentClip.size);

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

                if (currentFrame >= FFT_FRAMES && sinceLastFrame >= skipFrames) {
                    sinceLastFrame = 0;
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
                sinceLastFrame++;

                fftStart += FFT_FRAME_SPACING;
                currentFrame++;
            }
        }

        ens::StoreBestCoordinates<mat> bestCoordinates;
        network.Train(train,
            trainLabels,
            optimizer,
            ens::PrintLoss(),
            ens::ProgressBar(),
            ens::EarlyStopAtMinLoss(
                [&](const mat& /* param */)
                {
                    double validationLoss = network.Evaluate(test, testLabels);
                    cout << "Validation loss: " << validationLoss
                        << "." << endl;
                    return validationLoss;
                }),
            // Store best coordinates (neural network weights)
            bestCoordinates);
        optimizer.ResetPolicy() = false;

        network.Parameters() = bestCoordinates.BestCoordinates();

        ModelSerializer::save((void*)&network);
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

void PhonemeClassifier::processFrame(Frame& frame, const float* audio, const size_t& start, const size_t& totalSize) {
    float max = 0.0;
    for (size_t i = 0; i < FFT_FRAME_SAMPLES; i++) {
        max = fmaxf(max, abs(audio[i]));
    }
    frame.volume = max;
    float scale = 1.0f / fmaxf(0.01f, frame.volume);
    for (size_t i = 0; i < FFT_FRAME_SAMPLES; i++) {
        size_t readLocation = (start + i) % totalSize;
        const float& value = audio[readLocation];
        fftwIn[i] = value * scale * window[i];
    }
    fftwf_execute(fftwPlan);
    for (size_t i = 0; i < FFT_REAL_SAMPLES; i++) {
        fftwf_complex& complex = fftwOut[i];
        float amplitude = abs(complex[0]);
        float newDelta = (amplitude - frame.real[i]) * 10;
        newDelta = fmaxf(0, newDelta);
        frame.delta[i] = (newDelta + frame.delta[i] * 3) / 4;
        frame.real[i] = amplitude;
        frame.imaginary[i] = complex[1];
    }
}

void PhonemeClassifier::writeInput(const std::vector<Frame>& frames, const size_t& lastWritten, mat& data, size_t col) {
    for (size_t f = 0; f < FFT_FRAMES; f++) {
        const Frame& readFrame = frames[(lastWritten + frames.size() - f) % frames.size()];
        size_t offset = f * FFT_REAL_SAMPLES * 2;
        for (size_t i = 0; i < FFT_REAL_SAMPLES; i++) {
            data(offset + i, col) = readFrame.real[i];
            data(offset + FFT_REAL_SAMPLES + i, col) = readFrame.delta[i];
        }
    }
}

void PhonemeClassifier::preprocessDataset(const std::string& path) {
    // Generate phoneme list
    TSVReader dictReader;
    dictReader.open(path + "/english_us_mfa.dict");
    std::vector<std::string> phonemeList = std::vector<std::string>();
    while (dictReader.good()) {
        std::string* tabSeperated = dictReader.read_line();
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
        while (tsv.good()) {
            auto start = std::chrono::system_clock::now();
            // Batches
            int counter = 0;
            Clip clip = Clip();
            clip.initSampleRate(SAMPLE_RATE);
            while (tsv.good() && !(counter >= PREPROCESS_BATCH_SIZE && globalCounter % PREPROCESS_BATCH_SIZE == 0)) {
                std::cout << globalCounter << "\n";

                loadNextClip(clipPath, tsv, clip, -1);

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

void PhonemeClassifier::loadNextClip(const std::string& clipPath, TSVReader& tsv, OUT Clip& clip, int sampleRate) {
    clip.clipPath = clipPath;
    clip.loaded = false;
    std::string* elements = tsv.read_line();
    clip.tsvElements = elements;
    if (sampleRate > 0) {
        clip.loadMP3(sampleRate);
    }
}

void PhonemeClassifier::initalizePhonemeSet() {
#define REGISTER_PHONEME(t, p) \
    phonemeSet[PhonemeClassifier::customHasher(L##p)] = _phonemeCounter++; \
    inversePhonemeSet.push_back(t);

    size_t _phonemeCounter = 0;
    using namespace std::string_literals;
    REGISTER_PHONEME("", "")
        REGISTER_PHONEME("ai", "aj" )
        REGISTER_PHONEME("ow", "aw" )
        REGISTER_PHONEME("b" , "b"  )
        REGISTER_PHONEME("b" , "bʲ" )
        REGISTER_PHONEME("c" , "c"  )
        REGISTER_PHONEME("c" , "cʰ" )
        REGISTER_PHONEME("c" , "cʷ" )
        REGISTER_PHONEME("d" , "d"  )
        REGISTER_PHONEME("j" , "dʒ" )
        REGISTER_PHONEME("th", "dʲ" )
        REGISTER_PHONEME("du", "d̪" )
        REGISTER_PHONEME("e" , "ej" )
        REGISTER_PHONEME("f" , "f"  )
        REGISTER_PHONEME("f" , "fʲ" )
        REGISTER_PHONEME("h" , "h"  )
        REGISTER_PHONEME("i" , "i"  )
        REGISTER_PHONEME("i" , "iː" )
        REGISTER_PHONEME("j" , "j"  )
        REGISTER_PHONEME("k" , "k"  )
        REGISTER_PHONEME("k" , "kʰ" )
        REGISTER_PHONEME("k" , "kʷ" )
        REGISTER_PHONEME("l" , "l"  )
        REGISTER_PHONEME("m" , "m"  )
        REGISTER_PHONEME("m" , "mʲ" )
        REGISTER_PHONEME("m" , "m̩" )
        REGISTER_PHONEME("n" , "n"  )
        REGISTER_PHONEME("n" , "n̩" )
        REGISTER_PHONEME("o" , "ow" )
        REGISTER_PHONEME("p" , "p"  )
        REGISTER_PHONEME("p" , "pʰ" )
        REGISTER_PHONEME("p" , "pʲ" )
        REGISTER_PHONEME("p" , "pʷ" )
        REGISTER_PHONEME("s" , "s"  )
        REGISTER_PHONEME("=" , "spn")
        REGISTER_PHONEME("t" , "t"  )
        REGISTER_PHONEME("ch", "tʃ" )
        REGISTER_PHONEME("t" , "tʰ" )
        REGISTER_PHONEME("t" , "tʲ" )
        REGISTER_PHONEME("t" , "tʷ" )
        REGISTER_PHONEME("th", "t̪" )
        REGISTER_PHONEME("v" , "v"  )
        REGISTER_PHONEME("v" , "vʲ" )
        REGISTER_PHONEME("w" , "w"  )
        REGISTER_PHONEME("z" , "z"  )
        REGISTER_PHONEME("ae", "æ"  )
        REGISTER_PHONEME("s" , "ç"  )
        REGISTER_PHONEME("th", "ð"  )
        REGISTER_PHONEME("ng", "ŋ"  )
        REGISTER_PHONEME("a" , "ɐ"  )
        REGISTER_PHONEME("a" , "ɑ"  )
        REGISTER_PHONEME("a" , "ɑː" )
        REGISTER_PHONEME("a" , "ɒ"  )
        REGISTER_PHONEME("a" , "ɒː" )
        REGISTER_PHONEME("o" , "ɔj" )
        REGISTER_PHONEME("uh", "ə"  )
        REGISTER_PHONEME("ah", "ɚ"  )
        REGISTER_PHONEME("eh", "ɛ"  )
        REGISTER_PHONEME("uh", "ɝ"  )
        REGISTER_PHONEME("g" , "ɟ"  )
        REGISTER_PHONEME("g" , "ɟʷ" )
        REGISTER_PHONEME("g" , "ɡ"  )
        REGISTER_PHONEME("g" , "ɡʷ" )
        REGISTER_PHONEME("i" , "ɪ"  )
        REGISTER_PHONEME("l" , "ɫ"  )
        REGISTER_PHONEME("l" , "ɫ̩" )
        REGISTER_PHONEME("mf", "ɱ"  )
        REGISTER_PHONEME("ny", "ɲ"  )
        REGISTER_PHONEME("r" , "ɹ"  )
        REGISTER_PHONEME("tt", "ɾ"  )
        REGISTER_PHONEME("tt", "ɾʲ" )
        REGISTER_PHONEME("tt", "ɾ̃"  )
        REGISTER_PHONEME("sh", "ʃ"  )
        REGISTER_PHONEME("u" , "ʉ"  )
        REGISTER_PHONEME("u" , "ʉː" )
        REGISTER_PHONEME("oo", "ʊ"  )
        REGISTER_PHONEME("ll", "ʎ"  )
        REGISTER_PHONEME("z" , "ʒ"  )
        REGISTER_PHONEME("/" , "ʔ"  )
        REGISTER_PHONEME("th", "θ"  )
#undef REGISTER_PHONEME
}
