#include "Dataset.h"

#define DR_MP3_IMPLEMENTATION
#define DR_WAV_IMPLEMENTATION

#if 1
#define DO_NAN_CHECK
#endif

#include <algorithm>
#include <iostream>
#include <dr_mp3.h>
#include <dr_wav.h>
#include <samplerate.h>
#include "ClassifierHelper.h"
#include "Util.h"
#include "Global.h"

void Dataset::get(OUT CPU_MAT_TYPE& data, OUT CPU_MAT_TYPE& labels) {
    auto lock = sharedData.lock();
    size_t outputSize = sharedData.exampleData.size();
    size_t inputSize = sharedData.exampleData[0].n_rows;

    if (cached) {
        data = cachedData;
        labels = cachedLabels;
    } else {
        data = CPU_MAT_TYPE(inputSize, 0);
        labels = CPU_MAT_TYPE(1, 0);
        for (size_t i = 0; i < outputSize; i++) {
            size_t offset = sharedData.examples * i;
            const CPU_MAT_TYPE& dataMat = sharedData.exampleData.back();
            const CPU_MAT_TYPE& labelMat = sharedData.exampleLabel.back();
            data.resize(inputSize, data.n_cols + sharedData.examples);
            labels.resize(1, labels.n_cols + sharedData.examples);
            for (size_t c = 0; c < sharedData.examples; c++) {
                for (size_t r = 0; r < inputSize; r++) {
                    data(r, offset + c) = dataMat(r, c);
                }
                labels(0, offset + c) = i;
            }
            sharedData.exampleData.pop_back();
            sharedData.exampleLabel.pop_back();
        }
    }
}

void Dataset::start(size_t inputSize, size_t outputSize, size_t ex, bool print) {
    if (cached) {
        return;
    }
    ex = std::max((size_t)1, ex);

    // Set up shared data
    sharedData.exampleData = std::vector<CPU_MAT_TYPE>(outputSize);
    sharedData.exampleLabel = std::vector<CPU_MAT_TYPE>(outputSize);
	sharedData.exampleCount = std::vector<size_t>(outputSize);
    sharedData.examples = ex;
    for (size_t i = 0; i < outputSize; i++) {
        sharedData.exampleData[i] = CPU_MAT_TYPE(inputSize, ex);
        sharedData.exampleLabel[i] = CPU_MAT_TYPE(1, ex);
        sharedData.exampleLabel[i].fill(i);
        sharedData.exampleCount[i] = 0;
    }
    sharedData.minExamples = 0;
    sharedData.totalClips = 0;
    sharedData.testedClips = 0;
    sharedData.transcriptsPath = sharedData.path + "/transcript/";
    if (sharedData.type == Type::TIMIT)
        sharedData.timitIter.open(sharedData.path);
    sharedData.reader.shuffle();
    sharedData.reader.resetLine();
    
    // Start workers
    for (auto& worker : workers) {
        if (worker == nullptr)
            worker = std::make_shared<DatasetWorker>();
        worker->setData(sharedData);
        worker->doWork();
    }

    loaderThread = std::thread([this, inputSize, outputSize, ex, print] {
        _start(inputSize, outputSize, ex, print);
    });
}

bool Dataset::join() {
    if (cached) {
        return true;
    }
    if (loaderThread.joinable())
        loaderThread.join();

    return true;
}

bool Dataset::done() {
    for (auto& worker : workers) {
        if (!worker->done())
            return false;
    }
    return true;
}

void Dataset::end() {
    for (auto& worker : workers) {
        worker->end();
    }
}

void Dataset::setSubtype(Dataset::Subtype t) {
    auto lock = sharedData.lock();
    if (sharedData.type == COMMON_VOICE) {
        std::string tsv = sharedData.path;
        switch (t) {
        case TRAIN:
            tsv += "/train.tsv";
            break;
        case TEST:
            tsv += "/test.tsv";
            break;
        case VALIDATE:
            tsv += "/dev.tsv";
            break;
        }
        sharedData.reader.open(tsv, false, clientFilter);
    } else if (sharedData.type == TIMIT) {
        switch (t) {
        case TRAIN:
            sharedData.path += "/TRAIN";
            break;
        case TEST:
            sharedData.path += "/TEST";
            break;
        case VALIDATE:
            sharedData.path += "/TEST";
            break;
        }
    }
}

size_t Dataset::getMinCount() {
    auto lock = sharedData.lock();
    return sharedData.minExamples;
}

void Dataset::_start(size_t inputSize, size_t outputSize, size_t ex, bool print) {
    size_t realEx = ex;

    for (auto& worker : workers) {
        worker->join();
    }

    if (sharedData.examples < realEx) {
        saveCache();
    }

    std::printf("Finished loading clips from %s with %zd examples (actual: %zd)\n", sharedData.reader.path().c_str(), sharedData.minExamples, std::min(realEx, sharedData.minExamples));
    std::printf("Loaded from %zd clips considering %zd total\n", sharedData.totalClips, sharedData.testedClips);
}

std::vector<Phone> Dataset::parseTextgrid(const std::string& path, int sampleRate) {
    std::ifstream reader;
    reader.open(path);
    if (!reader.is_open()) {
        std::printf("Failed to open file: %s\n", path.c_str());
        return {};
    }

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

        p.phonetic = G_PS.fromString(text);
        phones[i] = p;
    }

    reader.close();

    return phones;
}

std::vector<Phone> Dataset::parseTIMIT(const std::string& path, int sampleRate) {
    std::ifstream phonemeReader;
    phonemeReader.open(path);

    std::string line;
    std::vector<Phone> tempPhones;
    while (true) {
        std::getline(phonemeReader, line);
        if (line == "")
            break;

        size_t s1 = line.find(' ');
        size_t s2 = line.find(' ', s1 + 1);

        std::string i1 = line.substr(0, s1).c_str();
        std::string i2 = line.substr(s1 + 1, s2 - s1 - 1).c_str();
        std::string i3 = line.substr(s2 + 1).c_str();

        Phone p = Phone();
        p.minIdx = std::stoull(i1);
        p.min = p.minIdx / 16000.0;
        p.maxIdx = std::stoull(i2);
        p.max = p.maxIdx / 16000.0;
        p.phonetic = G_PS.fromString(i3);

        tempPhones.push_back(std::move(p));
    };

    phonemeReader.close();
    return tempPhones;
}

void Dataset::loadNextClip(const std::string& clipPath, TSVReader::TSVLine tabSeperated, OUT Clip& clip, int sampleRate) {
    clip.tsvElements = tabSeperated;
    loadNextClip(clipPath, clip, sampleRate);
}

void Dataset::loadNextClip(const std::string& clipPath, TSVReader& tsv, OUT Clip& clip, int sampleRate) {
    TSVReader::TSVLine elements = TSVReader::convert(*tsv.read_line());
    loadNextClip(clipPath, elements, clip, sampleRate);
}

void Dataset::loadNextClip(const std::string& clipPath, OUT Clip& clip, int sampleRate) {
    clip.clipPath = clipPath;
    clip.loaded = false;
    if (sampleRate > 0) {
        clip.load(sampleRate);
    }
}

void Dataset::Clip::load(int targetSampleRate) {
    // Read mp3 file from tsv
    //printf("Loading MP3: %s\n", tsvElements[TSVReader::Indices::PATH].c_str());

    size_t clipSamples, clipSampleRate;
    float* floatBuffer = NULL;

    std::string clipFullPath = getFilePath();
    if (clipFullPath == "")
        return;

    switch (type) {
    case COMMON_VOICE:
        floatBuffer = loadMP3(clipSamples, clipSampleRate, clipFullPath);
        break;
    case TIMIT:
        floatBuffer = loadWAV(clipSamples, clipSampleRate, clipFullPath);
        break;
    }
    if (floatBuffer == NULL) {
        printf("Failed to open file: %s\n", clipFullPath.c_str());
        return;
    }

    float length = (double)clipSamples / clipSampleRate;
    float newLength = length + 0.1f;
    if (newLength > allocatedLength) {
        // Not very efficient but should slow/stop after a certain point
        delete[] buffer;
        buffer = new float[(size_t)(targetSampleRate * newLength)];
        allocatedLength = newLength;
    }

    // Convert sample rate
    if (clipSampleRate == targetSampleRate) {
        memcpy(buffer, floatBuffer, sizeof(float) * clipSamples);
        size = clipSamples;
        sampleRate = clipSampleRate;
    } else if (clipSampleRate % targetSampleRate == 0 && clipSampleRate > targetSampleRate) {
        // Integer down sample rate conversion
        int factor = clipSampleRate / targetSampleRate;
        size = clipSamples / factor;
        for (size_t i = 0; i < size; i++) {
            buffer[i] = floatBuffer[i * factor];
        }
        sampleRate = targetSampleRate;
    } else {
        SRC_DATA upsampleData = SRC_DATA();
        upsampleData.data_in = floatBuffer;
        upsampleData.input_frames = clipSamples;
        double ratio = (double)targetSampleRate / clipSampleRate;
        long outSize = (long)(ratio * clipSamples);
        upsampleData.src_ratio = ratio;
        upsampleData.data_out = buffer;
        upsampleData.output_frames = outSize;
        int error = src_simple(&upsampleData, SRC_SINC_BEST_QUALITY, 1);
        if (error) {
            std::cout << "Error while resampling: " << src_strerror(error) << '\n';
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
    float targetMax = 0.8f;
    float factor = targetMax / max;
    for (size_t i = 0; i < size; i++) {
        buffer[i] *= factor;
    }

    loaded = true;
}

float* Dataset::Clip::loadMP3(OUT size_t& samples, OUT size_t& sampleRate, const std::string& path) {
    drmp3_config cfg;
    drmp3_uint64 scount;
    float* floatBuffer = drmp3_open_file_and_read_pcm_frames_f32(path.c_str(), &cfg, &scount, NULL);

    if (cfg.channels <= 0) {
        printf("%s has invalid channel count (%d)\n", path.c_str(), cfg.channels);
        return NULL;
    }
    if (cfg.sampleRate <= 0) {
        printf("%s has invalid sample rate (%d)\n", path.c_str(), cfg.sampleRate);
        return NULL;
    }

    samples = scount;
    convertMono(floatBuffer, samples, cfg.channels);
    sampleRate = cfg.sampleRate;
    return floatBuffer;
}

float* Dataset::Clip::loadWAV(OUT size_t& samples, OUT size_t& sampleRate, const std::string& path) {
    unsigned int channels, sr;
    drwav_uint64 clipSamples;
    float* floatBuffer = drwav_open_file_and_read_pcm_frames_f32(path.c_str(), &channels, &sr, &clipSamples, NULL);

    if (channels <= 0) {
        printf("%s has invalid channel count (%d)\n", path.c_str(), channels);
        return NULL;
    }
    if (sr <= 0) {
        printf("%s has invalid sample rate (%d)\n", path.c_str(), sr);
        return NULL;
    }

    samples = clipSamples;
    convertMono(floatBuffer, samples, channels);
    sampleRate = sr;
    return floatBuffer;
}

std::string Dataset::Clip::getFilePath() {
    std::string path;
    switch (type) {
    case COMMON_VOICE:
        path = clipPath + tsvElements.PATH;
        break;
    case TIMIT:
        path = clipPath;
        break;
    default:
        throw("Invalid type");
        break;
    }
    if (!std::filesystem::exists(path)) {
        printf("%s does not exist\n", path.c_str());
        return "";
    }
    return path;
}

void Dataset::Clip::convertMono(float* buffer, OUT size_t& length, int channels) {
    if (channels <= 1)
        return;
    
    length /= channels;
    for (size_t i = 0; i < length; i++) {
        float sum = 0;
        size_t off = i * channels;
        for (int j = 0; j < channels; j++) {
            sum += buffer[off + j];
        }
        buffer[i] = sum / channels;
    }
}

void Dataset::preprocessDataset(const std::string& path, const std::string& workDir, const std::string& dictPath, const std::string& acousticPath, const std::string& outputDir, size_t batchSize) {
    auto lock = sharedData.lock();
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
        "/dev.tsv",
        "/test.tsv",
        "/train.tsv"
    };
    const std::string clipPath = path + "/clips/";
    for (int i = 0; i < tables.size(); i++) {
        TSVReader tsv;
        tsv.open((path + tables[i]).c_str(), true);

        unsigned long globalCounter = 0;
        bool tsvGood = true;
        while (tsvGood) {
            auto start = std::chrono::system_clock::now();
            // Batches
            int counter = 0;
            Clip clip = Clip();
            clip.type = COMMON_VOICE;
            clip.initSampleRate(sharedData.sampleRate);
            TSVReader::CompactTSVLine* compact = tsv.read_line();

            std::string audioWorkDir = workDir + "/audio";
            std::string mfaWorkDir = workDir + "/mfa";
            std::filesystem::create_directories(audioWorkDir);
            std::filesystem::create_directories(mfaWorkDir);

            while (compact != NULL && !(counter >= batchSize && globalCounter % batchSize == 0)) {
                TSVReader::TSVLine tabSeperated = TSVReader::convert(*compact);
                compact = tsv.read_line();
                std::cout << globalCounter << "\r";

                loadNextClip(clipPath, tabSeperated, clip, -1);

                globalCounter++;
                std::string transcriptPath = outputDir + "/" + clip.tsvElements.CLIENT_ID + "/" + clip.tsvElements.PATH;
                transcriptPath = transcriptPath.substr(0, transcriptPath.length() - 4) + ".TextGrid";
                if (std::filesystem::exists(transcriptPath)) {
                    continue;
                }
                clip.load(16000);
                if (!clip.loaded) {
                    continue;
                }

                std::string speakerPath = audioWorkDir + "/" + clip.tsvElements.CLIENT_ID + "/";
                if (!std::filesystem::exists(speakerPath)) {
                    std::filesystem::create_directory(speakerPath);
                }
                std::string originalPath = clip.tsvElements.PATH;
                std::string fileName = speakerPath + originalPath.substr(0, originalPath.length() - 4);

                // Convert audio to wav
                drwav wav;
                drwav_data_format format = drwav_data_format();
                format.container = drwav_container_riff;
                format.format = DR_WAVE_FORMAT_PCM;
                format.channels = 1;
                format.sampleRate = 16000;
                format.bitsPerSample = 16;
                drwav_init_file_write(&wav, (fileName + ".wav").c_str(), &format, NULL);

                drwav_int16* intBuffer = new drwav_int16[clip.size];
                for (size_t j = 0; j < clip.size; j++) {
                    intBuffer[j] = std::numeric_limits<drwav_int16>::max() * clip.buffer[j];
                }
                drwav_uint64 framesWritten = drwav_write_pcm_frames(&wav, clip.size, intBuffer);
                delete[] intBuffer;

                drwav_uninit(&wav);

                // Transcript
                std::ofstream fstream;
                fstream.open(fileName + ".txt");
                std::string sentence = clip.tsvElements.SENTENCE;
                fstream.write(sentence.c_str(), sentence.length());
                fstream.close();

                counter++;
            }
            std::cout << "\n";
            // Run alignment
            const std::string mfaLine = std::format("mfa align -t {} --use_mp --quiet --clean {} {} {} {}", mfaWorkDir, audioWorkDir, dictPath, acousticPath, outputDir);
            int result = system(mfaLine.c_str());
            if (result != 0)
                std::printf("MFA exited with exit code %d\n", result);
            // Cleanup
            std::filesystem::directory_iterator iterator(audioWorkDir);
            for (const auto& directory : iterator) {
                std::filesystem::remove_all(directory);
            }
            auto diff = std::chrono::system_clock::now() - start;
            double duration = std::chrono::duration_cast<std::chrono::duration<double>>(diff).count();
            std::cout << "Iteration took " << duration << " seconds\n";
            tsvGood = compact != NULL;
        }
    }
}

std::vector<float> Dataset::_findAndLoad(const std::string& path, size_t target, int samplerate, TSVReader::TSVLine& tsv, std::vector<Phone>& phones, const std::string& filter) {
    auto lock = sharedData.lock();
    sharedData.reader.shuffle();
    const std::string clipPath = path + "/clips/";
    const std::string transcriptsPath = path + "/transcript/";
    Clip clip;
    clip.type = COMMON_VOICE;
    clip.initSampleRate(samplerate);
    std::vector<float> audio = std::vector<float>();
    TSVReader::CompactTSVLine* line;
    while (true) {
        TSVReader::CompactTSVLine* line = sharedData.reader.read_line();
        if (!line) {
            sharedData.reader.resetLine();
            break;
        }
       tsv = TSVReader::convert(*line);
        if (filter != "" && tsv.CLIENT_ID != filter)
            continue;
        // Get transcription
        const std::string& nextClipPath = tsv.PATH;
        const std::string transcriptionPath = transcriptsPath + tsv.CLIENT_ID + "/" + nextClipPath.substr(0, nextClipPath.length() - 4) + ".TextGrid";
        if (!std::filesystem::exists(transcriptionPath)) {
            //std::cout << "Missing transcription: " << transcriptionPath << std::endl;
            continue;
        }
        phones = parseTextgrid(transcriptionPath, sharedData.sampleRate);

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

inline bool Dataset::keepLoading(size_t minExamples, DatasetWorker::DatasetData& data, bool endFlag) {
    auto lock = data.lock();
    return (minExamples < data.examples || !endFlag) && 
            (minExamples < data.examples * MMAX_EXAMPLE_F); 
}

void Dataset::saveCache() {
    get(cachedData, cachedLabels);
    cached = true;
}

bool Dataset::frameHasNan(const Frame& frame) {
#ifdef DO_NAN_CHECK
    for (int i = 0; i < FRAME_SIZE; i++) {
        if (!(std::isfinite(frame.avg[i]) && 
              std::isfinite(frame.delta[i]) && 
              std::isfinite(frame.accel[i])))
            return true;
    }
#endif
    return false;
}

bool Dataset::wantClip(const std::vector<Phone>& phones, const std::vector<size_t>& exampleCount, size_t examples) {
    bool shouldLoad = false;
    for (size_t i = 0; i < phones.size(); i++) {
        size_t binCount = exampleCount[phones[i].phonetic];
        if (binCount < examples * MMAX_EXAMPLE_F) {
            shouldLoad = true;
            break;
        }
    }
    return shouldLoad;
}

void Dataset::DatasetWorker::work(SharedData* _d) {
    DatasetData& data = *(DatasetData*)_d;

    Dataset::Type type;
    Clip clip;
    int sampleRate;
    std::string path, clipPath;
    {
        auto lock = data.lock();
        type = data.type;
        sampleRate = data.sampleRate;
        clip.initSampleRate(data.sampleRate);
        clip.type = type;
        path = data.path;
        clipPath = data.path + "/clips/";
    }
    TSVReader::CompactTSVLine* line;
    TSVReader::TSVLine nextClip;
    
    std::vector<Phone> phones;
    std::vector<Frame> frames;
    ClassifierHelper helper;
    helper.initalize(sampleRate);
    while (Dataset::keepLoading(data.minExamples, data, _end)) {
        // Find a clip
        {
            auto lock = data.lock();
            if (type == COMMON_VOICE) {
                line = data.reader.read_line(data.lineIndex);
                if (!line) {
                    break;
                }
                nextClip = TSVReader::convert(*line);
                // Get transcription
                const std::string& nextClipPath = nextClip.PATH;
                const std::string transcriptionPath = data.transcriptsPath + nextClip.CLIENT_ID + "/" + nextClipPath.substr(0, nextClipPath.length() - 4) + ".TextGrid";
                if (!std::filesystem::exists(transcriptionPath)) {
                    //std::cout << "Missing transcription: " << transcriptionPath << std::endl;
                    data.reader.dropIdx(data.lineIndex);
                    continue;
                }
                std::vector<Phone> tempPhones = Dataset::parseTextgrid(transcriptionPath, sampleRate);

                // Check if the clip should be loaded
                bool shouldLoad = Dataset::wantClip(tempPhones, data.exampleCount, data.examples);
                data.testedClips++;
                if (!shouldLoad) {
                    continue;
                }
                phones = std::move(tempPhones);
                loadNextClip(clipPath, nextClip, clip, -1);
            } else if (type == TIMIT) {
                if (!data.timitIter.good())
                    break;
                auto path = data.timitIter.next();
                auto fname = path.filename();
                std::vector<Phone> tempPhones = parseTIMIT(path.string(), sampleRate);
                data.testedClips++;
                bool shouldLoad = wantClip(tempPhones, data.exampleCount, data.examples);
                std::string cpath = path.string();
                cpath = cpath.substr(0, cpath.size() - 4);
                cpath += "_.wav";

                if (!shouldLoad)
                    continue;

                phones = std::move(tempPhones);
                loadNextClip(cpath, clip, -1);
            }
        }

        // Load the audio
        clip.load(sampleRate);
        for (Frame& frame : frames) {
            frame.reset();
        }
        if (clip.loaded) {
            data.totalClips++;

            // Convert to data used by classifier
            size_t frameCounter = 0;
            {
                size_t fftStart = 0;
                // Load all frames
                while (fftStart + FFT_REAL_SAMPLES < clip.size) {
                    if (frames.size() <= frameCounter) {
                        frames.emplace_back();
                        Frame& newFrame = frames.back();
                        newFrame.reset();
                    }
                    Frame& currentFrame = frames[frameCounter];
                    helper.processFrame(clip.buffer, fftStart, clip.size, frames, frameCounter);
                    currentFrame.invalid = Dataset::frameHasNan(currentFrame);

                    size_t maxOverlap = 0;
                    size_t maxIdx = 0;
                    // Assign phoneme ID to frame
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

            {
                auto lock = data.lock();
                for (size_t i = FFT_FRAMES; i < frameCounter; i++) {
                    Frame& frame = frames[i - CONTEXT_BACKWARD];
                    const size_t& currentPhone = frame.phone;
                    auto& phonemeCounter = data.exampleCount[currentPhone];

                    // Write data
                    size_t writeCol;
                    if (phonemeCounter < data.examples) {
                        writeCol = phonemeCounter;
                    } else {
                        double rnd = (double)rand() / RAND_MAX;
                        double flr = (double)data.examples / phonemeCounter;
                        if (rnd < flr) {
                            writeCol = rand() % data.examples;
                        } else {
                            continue;
                        }
                    }
                    if (helper.writeInput<CPU_MAT_TYPE>(frames, i, data.exampleData[currentPhone], writeCol)) {
                        phonemeCounter++;
                    }
                }

                // Count how many examples have been collected
                data.minExamples = data.exampleCount[0];
                size_t minIdx = 0;
                for (size_t i = 1; i < data.exampleCount.size(); i++) {
                    if (data.exampleCount[i] < data.minExamples) {
                        data.minExamples = data.exampleCount[i];
                        minIdx = i;
                    }
                }
            }

            /*
            minExamples = minTemp;
            if (print && endFlag) {
                std::printf("%d clips; Min: %d of %s\r", (int)totalClips, (int)minExamples, G_PS.xSampa(minIdx).c_str());
                fflush(stdout);
            }
            */
        } else { // Could not load
            if (type == COMMON_VOICE) {
                auto lock = data.lock();
                data.reader.dropIdx(data.lineIndex);
            }
        }
    }
}
