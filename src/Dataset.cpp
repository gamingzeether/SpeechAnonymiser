#include "Dataset.h"

#define DR_MP3_IMPLEMENTATION
#define DR_WAV_IMPLEMENTATION

#ifdef __GNUC__
// Fix an issue where some frames have nan values
// Seems to come from the dct returning nan but not sure why
// Doesnt seem to happen on x64 msvc143 but does occasionally on aarch64 gcc13
#define DO_NAN_CHECK
#endif

#include <algorithm>
#include <filesystem>
#include <thread>
#include <iostream>
#include <condition_variable>
#include <dr_mp3.h>
#include <dr_wav.h>
#include <samplerate.h>
#include "ClassifierHelper.h"
#include "Util.h"
#include "Global.h"

void Dataset::get(OUT CPU_MAT_TYPE& data, OUT CPU_MAT_TYPE& labels, bool destroy) {
    size_t outputSize = exampleData.size();
    size_t inputSize = exampleData[0].n_rows;

    if (cached) {
        data = cachedData;
        labels = cachedLabels;
    } else {
        data = CPU_MAT_TYPE(inputSize, 0);
        labels = CPU_MAT_TYPE(1, 0);
        for (size_t i = 0; i < outputSize; i++) {
            size_t offset = examples * i;
            const CPU_MAT_TYPE& dataMat = exampleData.back();
            const CPU_MAT_TYPE& labelMat = exampleLabel.back();
            data.resize(inputSize, data.n_cols + examples);
            labels.resize(1, labels.n_cols + examples);
            for (size_t c = 0; c < examples; c++ /* <-- he said the name of the movie! */) {
                for (size_t r = 0; r < inputSize; r++) {
                    data(r, offset + c) = dataMat(r, c);
                }
                labels(0, offset + c) = labelMat(0, c);
            }
            if (destroy) {
                exampleData.pop_back();
                exampleLabel.pop_back();
            }
        }
    }
}

void Dataset::start(size_t inputSize, size_t outputSize, size_t ex, bool print) {
    if (cached) {
        return;
    }
    endFlag = false;
    ex = std::max((size_t)1, ex);
    loaderThread = std::thread([this, inputSize, outputSize, ex, print] {
        _start(inputSize, outputSize, ex, print);
    });
}

bool Dataset::join() {
    if (cached) {
        return true;
    }
    if (!loaderThread.joinable()) {
        return false;
    }
    endFlag = true;
    loaderThread.join();
    return true;
}

void Dataset::_start(size_t inputSize, size_t outputSize, size_t ex, bool print) {
    reader.shuffle();
    reader.resetLine();

	std::vector<size_t> exampleCount(outputSize);
	exampleData = std::vector<CPU_MAT_TYPE>(outputSize);
    exampleLabel = std::vector<CPU_MAT_TYPE>(outputSize);

    const std::string clipPath = path + "/clips/";
    const std::string transcriptsPath = path + "/transcript/";

    examples = ex;
    for (size_t i = 0; i < outputSize; i++) {
        exampleData[i] = CPU_MAT_TYPE(inputSize, examples);
        exampleLabel[i] = CPU_MAT_TYPE(1, examples);
        exampleLabel[i].fill(i);
        exampleCount[i] = 0;
    }

    size_t minExamples = 0;
    Clip clip = Clip();
    clip.initSampleRate(sampleRate);
    std::vector<Phone> phones;
    std::mutex loaderMutex;
    std::condition_variable loaderWaiter;
    bool loaderReady = false;
    bool loaderFinished = true;
    size_t totalClips = 0;
    size_t lineIndex;
    size_t realEx = examples;

    // Load and process clip in seperate thread
    // Allow searching for next clip while current clip is being loaded
    // Find clip -> Find clip -> Find clip -> etc
    //  \-> Load clip -\-> Load clip
#pragma region MP3 Loader thread
    std::thread loaderThread = std::thread(
        [this, &minExamples, &inputSize, &outputSize, &print, &exampleCount,
        &loaderMutex, &loaderWaiter, &loaderReady,
        &clip, &loaderFinished, &phones, &totalClips, &lineIndex, &realEx] {
            std::vector<Frame> frames = std::vector <Frame>();
            while (keepLoading(minExamples, examples)) {
                {
                    std::unique_lock<std::mutex> lock(loaderMutex);
                    loaderWaiter.wait(lock, [&loaderReady] { return loaderReady; });
                    loaderReady = false;
                }

                clip.loadMP3(sampleRate);
                if (clip.loaded) {
                    totalClips++;

                    size_t frameCounter = 0;
                    {
                        size_t fftStart = 0;
                        // Load all frames
                        while (fftStart + FFT_REAL_SAMPLES < clip.size) {
                            if (frames.size() <= frameCounter) {
                                frames.push_back(Frame());
                            }
                            Frame& currentFrame = frames[frameCounter];
                            currentFrame.reset();
                            ClassifierHelper::instance().processFrame(clip.buffer, fftStart, clip.size, frames, frameCounter);
                            currentFrame.invalid = frameHasNan(currentFrame);

                            size_t maxOverlap = 0;
                            size_t maxIdx = 0;
                            // Look for a phoneme we need
                            for (int i = 0; i < phones.size(); i++) {
                                const Phone& p = phones[i];

                                if (p.maxIdx >= fftStart && fftStart >= p.minIdx) {
                                    currentFrame.phone = p.phonetic;
                                    break;
                                }
                            }

                            fftStart += FFT_FRAME_SPACING;
                            frameCounter++;
                        }
                    }

                    for (size_t i = 0; i < frameCounter; i++) {
                        Frame& frame = frames[i];
                        // NAN check
                        const size_t& currentPhone = frame.phone;
                        auto& phonemeCounter = exampleCount[currentPhone];

                        // Write data
                        if (phonemeCounter < examples) {
                            if (ClassifierHelper::instance().writeInput<CPU_MAT_TYPE>(frames, 0, exampleData[currentPhone], phonemeCounter))
                                phonemeCounter++;
                        } else {
                            double rnd = (double)rand() / RAND_MAX;
                            double flr = (double)examples / phonemeCounter;
                            if (rnd < flr) {
                                if (ClassifierHelper::instance().writeInput<CPU_MAT_TYPE>(frames, 0, exampleData[currentPhone], rand() % examples))
                                    phonemeCounter++;
                            }
                        }
                    }

                    // Count how many examples have been collected
                    size_t minTemp = exampleCount[0];
                    size_t minIdx = 0;
                    for (size_t i = 1; i < outputSize; i++) {
                        if (exampleCount[i] < minTemp) {
                            minTemp = exampleCount[i];
                            minIdx = i;
                        }
                    }
                    minExamples = minTemp;
                    if (print && endFlag) {
                        std::printf("%d clips; Min: %d of %s\r", (int)totalClips, (int)minExamples, Global::get().phonemeSet().xSampa(minIdx).c_str());
                        fflush(stdout);
                    }
                } else { // Could not load
                    reader.dropIdx(lineIndex);
                }

                {
                    std::lock_guard<std::mutex> lock(loaderMutex);
                    loaderFinished = true;
                    loaderWaiter.notify_one();
                }
            }
        });
#pragma endregion

    // Find clip with wanted phonemes
#pragma region Find clips
    size_t testedClips = 0;
    TSVReader::TSVLine nextClip;
    while (keepLoading(minExamples, examples)) {
        TSVReader::CompactTSVLine* line = reader.read_line(lineIndex);
        if (!line) {
            examples = std::min(minExamples, examples);
            break;
        }
        nextClip = TSVReader::convert(*line);
        // Get transcription
        const std::string& nextClipPath = nextClip.PATH;
        const std::string transcriptionPath = transcriptsPath + nextClip.CLIENT_ID + "/" + nextClipPath.substr(0, nextClipPath.length() - 4) + ".TextGrid";
        if (!std::filesystem::exists(transcriptionPath)) {
            //std::cout << "Missing transcription: " << transcriptionPath << std::endl;
            reader.dropIdx(lineIndex);
            continue;
        }
        std::vector tempPhones = parseTextgrid(transcriptionPath);

        // Check if the clip should be loaded
        bool shouldLoad = false;
        for (size_t i = 0; i < tempPhones.size(); i++) {
            size_t binCount = exampleCount[tempPhones[i].phonetic];
            if (binCount < examples * MMAX_EXAMPLE_F) {
                shouldLoad = true;
                break;
            }
        }
        testedClips++;
        if (!shouldLoad) {
            continue;
        }

        {
            std::unique_lock<std::mutex> lock(loaderMutex);
            loaderWaiter.wait(lock, [&loaderFinished] { return loaderFinished; });
            loaderFinished = false;
        }

        phones = tempPhones;
        loadNextClip(clipPath, nextClip, clip, -1);

        {
            std::lock_guard<std::mutex> lock(loaderMutex);
            loaderReady = true;
            loaderWaiter.notify_one();
        }
    }
#pragma endregion

    endFlag = true;
    {
        std::lock_guard<std::mutex> lock(loaderMutex);
        loaderReady = true;
        loaderWaiter.notify_one();
    }
    loaderThread.join();
    if (examples < realEx) {
        saveCache();
    }

    std::printf("Finished loading clips from %s with %zd examples (actual: %zd)\n", reader.path().c_str(), minExamples, std::min(realEx, minExamples));
    std::printf("Loaded from %zd clips considering %zd total\n", totalClips, testedClips);
}

std::vector<Phone> Dataset::parseTextgrid(const std::string& path) {
    std::ifstream reader;
    reader.open(path);

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
    std::vector<Phone> phones(size);
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
        p.minIdx = sampleRate * p.min;
        p.maxIdx = sampleRate * p.max;

        p.phonetic = Global::get().phonemeSet().fromString(text);
        phones[i] = p;
    }

    reader.close();

    return phones;
}

void Dataset::loadNextClip(const std::string& clipPath, TSVReader::TSVLine tabSeperated, OUT Clip& clip, int sampleRate) {
    clip.clipPath = clipPath;
    clip.loaded = false;
    clip.tsvElements = tabSeperated;
    if (sampleRate > 0) {
        clip.loadMP3(sampleRate);
    }
}

void Dataset::loadNextClip(const std::string& clipPath, TSVReader& tsv, OUT Clip& clip, int sampleRate) {
    TSVReader::TSVLine elements = TSVReader::convert(*tsv.read_line());
    loadNextClip(clipPath, elements, clip, sampleRate);
}

void Dataset::Clip::loadMP3(int targetSampleRate) {
    // Read mp3 file from tsv
    //printf("Loading MP3: %s\n", tsvElements[TSVReader::Indices::PATH].c_str());

    drmp3_config cfg;
    drmp3_uint64 samples;

    std::string clipFullPath = clipPath + tsvElements.PATH;
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

void Dataset::preprocessDataset(const std::string& path, const std::string& workDir, const std::string& dictPath, const std::string& acousticPath, const std::string& outputDir) {
    // Generate phoneme list
    /*
    TSVReader dictReader;
    dictReader.open(path + "/english_us_mfa.dict");
    std::vector<std::string> phonemeList = std::vector<std::string>();
    TSVReader::TSVLine tabSeperated = TSVReader::convert(*dictReader.read_line());
    while (tabSeperated.CLIENT_ID != "") {
        std::string& phonemes = tabSeperated.PATH; // TSV is not dataset, get index 1
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
        tabSeperated = TSVReader::convert(*dictReader.read_line());
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
    */

    // Generate phoneme alignments
    const std::vector<std::string> tables = {
        "/train.tsv",
        "/dev.tsv",
        "/test.tsv"
    };
    const std::string clipPath = path + "/clips/";
    for (int i = 0; i < tables.size(); i++) {
        TSVReader tsv;
        tsv.open((path + tables[i]).c_str());

        unsigned long globalCounter = 0;
        bool tsvGood = true;
        while (tsvGood) {
            auto start = std::chrono::system_clock::now();
            // Batches
            int counter = 0;
            Clip clip = Clip();
            clip.initSampleRate(sampleRate);
            TSVReader::TSVLine tabSeperated = TSVReader::convert(*tsv.read_line());
            while (tabSeperated.CLIENT_ID != "" && !(counter >= PREPROCESS_BATCH_SIZE && globalCounter % PREPROCESS_BATCH_SIZE == 0)) {
                std::cout << globalCounter << "\n";

                loadNextClip(clipPath, tabSeperated, clip, -1);

                globalCounter++;
                std::string transcriptPath = path + "/transcript/" + clip.tsvElements.CLIENT_ID + "/" + clip.tsvElements.PATH;
                transcriptPath = transcriptPath.substr(0, transcriptPath.length() - 4) + ".TextGrid";
                if (std::filesystem::exists(transcriptPath)) {
                    continue;
                }
                clip.loadMP3(16000);

                std::string speakerPath = workDir + clip.tsvElements.CLIENT_ID + "/";
                if (!std::filesystem::is_directory(speakerPath)) {
                    std::filesystem::create_directory(speakerPath);
                }
                std::string originalPath = clip.tsvElements.PATH;
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
                std::string sentence = clip.tsvElements.SENTENCE;
                fstream.write(sentence.c_str(), sentence.length());
                fstream.close();

                counter++;
            }
            // Run alignment
            const std::string mfaLine = std::format("conda activate aligner && mfa align --clean {} {} {} {}", workDir, dictPath, acousticPath, outputDir);
            system(mfaLine.c_str());
            // Cleanup
            std::filesystem::directory_iterator iterator(workDir);
            for (const auto& directory : iterator) {
                std::filesystem::remove_all(directory);
            }
            auto diff = std::chrono::system_clock::now() - start;
            double duration = std::chrono::duration_cast<std::chrono::duration<double>>(diff).count();
            std::cout << "Iteration took " << duration << " seconds\n";
        }
    }
}

std::vector<float> Dataset::_findAndLoad(const std::string& path, size_t target, int samplerate, TSVReader::TSVLine& tsv, std::vector<Phone>& phones) {
    reader.shuffle();
    const std::string clipPath = path + "/clips/";
    const std::string transcriptsPath = path + "/transcript/";
    Clip clip;
    clip.initSampleRate(samplerate);
    std::vector<float> audio = std::vector<float>();
    TSVReader::CompactTSVLine* line;
    while (true) {
        TSVReader::CompactTSVLine* line = reader.read_line();
        if (!line) {
            reader.resetLine();
            break;
        }
       tsv = TSVReader::convert(*line);
        // Get transcription
        const std::string& nextClipPath = tsv.PATH;
        const std::string transcriptionPath = transcriptsPath + tsv.CLIENT_ID + "/" + nextClipPath.substr(0, nextClipPath.length() - 4) + ".TextGrid";
        if (!std::filesystem::exists(transcriptionPath)) {
            //std::cout << "Missing transcription: " << transcriptionPath << std::endl;
            continue;
        }
        phones = parseTextgrid(transcriptionPath);

        // Check if the clip should be loaded
        bool shouldLoad = false;
        for (size_t i = 0; i < phones.size(); i++) {
            size_t phoneme = phones[i].phonetic;
            if (phoneme == target) {
                shouldLoad = true;
                break;
            }
        }
        if (!shouldLoad) {
            continue;
        }
        loadNextClip(clipPath, tsv, clip, samplerate);
        audio.resize(clip.size);
        for (size_t i = 0; i < clip.size; i++) {
            audio[i] = clip.buffer[i];
        }
        return audio;
    }
    return audio;
}

void Dataset::saveCache() {
    get(cachedData, cachedLabels);
    cached = true;
}

bool Dataset::frameHasNan(const Frame& frame) {
#ifdef DO_NAN_CHECK
    for (int i = 0; i < FRAME_SIZE; i++)
        if (std::isnan(frame.avg[i]))
            return true;
#endif
    return false;
}
