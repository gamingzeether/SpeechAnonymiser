#include "Dataset.hpp"

#if 1
#define DO_NAN_CHECK
#endif

#include <algorithm>
#include <iostream>
#include <assert.h>
#include <dr_wav.h>
#include <mlpack/core/math/shuffle_data.hpp>
#include "../../Utils/Util.hpp"

#define CLIP_DURATION 8 // Max clip Duration
// Return codes for loading clips
#define CLIP_DONE -1
#define CLIP_GOOD 0
#define CLIP_SKIP 1

void Dataset::get(OUT CPU_CUBE_TYPE& data, OUT CPU_CUBE_TYPE& labels, arma::urowvec& sequenceLengths) {
    if (!done())
        return;
    auto lock = sharedData.lock();
    size_t outputSize = sharedData.exampleData.size();
    size_t inputSize = sharedData.exampleData.n_rows;

    data = std::move(sharedData.exampleData);
    labels = std::move(sharedData.exampleLabel);
    sequenceLengths = std::move(sharedData.sequenceLengths);
    mlpack::ShuffleData(data, labels, data, labels);
}

void Dataset::start(size_t inputSize, size_t outputSize, size_t ex, size_t batchSize, bool print) {
    ex = std::max((size_t)1, ex);

    // Set up shared data
    size_t audioSamples = sharedData.sampleRate * CLIP_DURATION;
    size_t nSlices = (audioSamples) / FFT_FRAME_SPACING;
    sharedData.nSlices = nSlices;
    sharedData.sequenceLengths.zeros(ex);
    sharedData.exampleData = CPU_CUBE_TYPE(inputSize, ex, nSlices);
    sharedData.exampleLabel = CPU_CUBE_TYPE(1, ex, nSlices);
    sharedData.exampleLabel.fill(0);
    sharedData.targetClips = ex;
    sharedData.totalClips = 0;
    sharedData.testedClips = 0;
    sharedData.transcriptsPath = sharedData.path + "/transcript/";
    sharedData.reader.shuffle();
    sharedData.reader.resetLine();
    
    // Start workers
    for (auto& worker : workers) {
        if (worker == nullptr)
            worker = std::make_shared<DatasetWorker>();
        worker->setData(sharedData);
        worker->doWork();
    }

    loaderThread = std::thread([this, inputSize, outputSize, ex, batchSize, print] {
        _start(inputSize, outputSize, ex, batchSize, print);
    });
}

bool Dataset::join() {
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

void Dataset::setSubtype(Subtype t) {
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

size_t Dataset::getLoadedClips() {
    auto lock = sharedData.lock();
    return sharedData.totalClips;
}

Type Dataset::folderType(const std::string& path) {
    if (std::filesystem::exists(path + "/train.tsv")) {
        return Type::COMMON_VOICE;
    } else {
        return Type::TIMIT;
    }
}

void Dataset::_start(size_t inputSize, size_t outputSize, size_t ex, size_t batchSize, bool print) {
    size_t realEx = ex;

    for (auto& worker : workers) {
        worker->join();
    }
    ex = std::min(realEx, sharedData.totalClips);
    size_t steps = ex / batchSize;
    assert(steps > 0);
    ex = batchSize * steps;
    sharedData.totalClips = ex;

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

        p.phonetic = G_PS_C.fromString(text);
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
        p.phonetic = G_PS_C.fromString(i3);

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

void Dataset::preprocessDataset(const std::string& path, const std::string& workDir, const std::string& dictPath, const std::string& acousticPath, const std::string& outputDir, size_t batchSize) {
    auto lock = sharedData.lock();

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
            clip.init(sharedData.sampleRate, CLIP_DURATION);
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

std::vector<float> Dataset::_findAndLoad(const std::string& path, size_t target, int samplerate, std::string& fileName, std::vector<Phone>& phones, const std::string& filter) {
    Clip clip;
    clip.type = sharedData.type;
    clip.init(samplerate, CLIP_DURATION);
    bool shouldLoad = false;
    // Only set if type is common voice
    TSVReader::TSVLine clipTsv;
    while (!shouldLoad) {
        // Load a clip
        int returnCode;
        switch (sharedData.type) {
            case COMMON_VOICE:
                returnCode = getNextClipCV(clip, sharedData, phones, clipTsv);
                if (returnCode == CLIP_DONE) {
                    sharedData.reader.shuffle();
                    sharedData.reader.resetLine();
                }
                break;
            case TIMIT:
                returnCode = getNextClipTIMIT(clip, sharedData, phones);
                if (returnCode == CLIP_DONE) {
                    sharedData.timitIter.resetCounter();
                }
                break;
        }

        // Check if it matches filter
        if (returnCode != CLIP_GOOD || (sharedData.type == COMMON_VOICE && filter != "" && clipTsv.CLIENT_ID != filter)) {
            continue;
        }
        
        // Check if clip contains the desired phone
        for (size_t i = 0; i < phones.size(); i++) {
            size_t phoneme = phones[i].phonetic;
            if (phoneme == target) {
                shouldLoad = true;
                break;
            }
        }
    }

    // Load the clip and return the audio
    clip.load(samplerate);
    std::vector<float> audio(clip.size);
    for (size_t i = 0; i < clip.size; i++) {
        audio[i] = clip.buffer[i];
    }
    return audio;
}

bool Dataset::clipTooLong(const std::vector<Phone>& phones) {
    return phones.back().max > CLIP_DURATION;
}

bool Dataset::keepLoading(DatasetWorker::DatasetData& data, bool endFlag) {
    auto lock = data.lock();
    return (data.totalClips < data.targetClips && !endFlag);
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

void Dataset::DatasetWorker::work(SharedData* _d) {
    DatasetData& data = *(DatasetData*)_d;

    Type type;
    Clip clip;
    int sampleRate;
    {
        auto lock = data.lock();
        type = data.type;
        sampleRate = data.sampleRate;
        clip.init(data.sampleRate, CLIP_DURATION);
        clip.type = type;
    }
    
    std::vector<Phone> phones;
    std::vector<Frame> frames;
    ClassifierHelper helper;
    helper.initalize(sampleRate);
    while (Dataset::keepLoading(data, _end)) {
        // Find a clip
        {
            int returnCode;
            switch (type) {
                case COMMON_VOICE:
                    returnCode = getNextClipCV(clip, data, phones);
                    break;
                case TIMIT:
                    returnCode = getNextClipTIMIT(clip, data, phones);
                    break;
            }
            if (returnCode == CLIP_DONE) {
                break;
            } else if (returnCode == CLIP_SKIP) {
                continue;
            }
        }

        // Load the audio
        clip.load(sampleRate);
        for (Frame& frame : frames) {
            frame.reset();
        }
        if (clip.loaded) {
            // Convert to data used by classifier
            size_t nFrames = 0;
            // Load all frames
            clipToFrames(clip, nFrames, frames, helper, phones);

            // Write data into matrix
            {
                auto lock = data.lock();
                size_t writeCol = data.totalClips;
                if (writeCol >= data.exampleData.n_cols)
                    writeCol = rand() % data.exampleData.n_cols;
                size_t nSlices = 0;
                for (size_t i = FFT_FRAMES; i < nFrames; i++) {
                    Frame& frame = frames[i - CONTEXT_BACKWARD];

                    // Write data
                    helper.writeInput(frames, i, data.exampleData, writeCol, nSlices);
                    data.exampleLabel(0, writeCol, nSlices) = frame.phone;
                    
                    nSlices++;
                }
                data.sequenceLengths(writeCol) = nSlices;
                data.totalClips++;
            }
        } else { // Could not load
            if (type == COMMON_VOICE) {
                auto lock = data.lock();
                data.reader.dropIdx(data.lineIndex);
            }
        }
    }
}

int Dataset::getNextClipCV(Clip& clip, Dataset::DatasetWorker::DatasetData& data, std::vector<Phone>& phones) {
    TSVReader::TSVLine dummyTsv;
    return getNextClipCV(clip, data, phones, dummyTsv);
}

int Dataset::getNextClipCV(Clip& clip, Dataset::DatasetWorker::DatasetData& data, std::vector<Phone>& phones, TSVReader::TSVLine& clipTsv) {
    auto lock = data.lock();
    TSVReader::CompactTSVLine* line;
    line = data.reader.read_line(data.lineIndex);
    if (!line) {
        return CLIP_DONE;
    }
    clipTsv = TSVReader::convert(*line);
    // Get transcription
    const std::string& nextClipPath = clipTsv.PATH;
    const std::string transcriptionPath = data.transcriptsPath + clipTsv.CLIENT_ID + "/" + nextClipPath.substr(0, nextClipPath.length() - 4) + ".TextGrid";
    if (!std::filesystem::exists(transcriptionPath)) {
        //std::cout << "Missing transcription: " << transcriptionPath << std::endl;
        data.reader.dropIdx(data.lineIndex);
        return CLIP_SKIP;
    }
    std::vector<Phone> tempPhones = Dataset::parseTextgrid(transcriptionPath, data.sampleRate);
    if (Dataset::clipTooLong(tempPhones))
        return CLIP_SKIP;

    data.testedClips++;
    phones = std::move(tempPhones);
    loadNextClip(data.path + "/clips/", clipTsv, clip, -1);
    
    return CLIP_GOOD;
}

int Dataset::getNextClipTIMIT(Clip& clip, Dataset::DatasetWorker::DatasetData& data, std::vector<Phone>& phones) {
    auto lock = data.lock();
    if (!data.timitIter.good())
        return CLIP_DONE;
    auto path = data.timitIter.next();
    auto fname = path.filename();
    std::vector<Phone> tempPhones = Dataset::parseTIMIT(path.string(), data.sampleRate);
    if (Dataset::clipTooLong(tempPhones))
        return CLIP_SKIP;
    data.testedClips++;
    std::string cpath = path.string();
    cpath = cpath.substr(0, cpath.size() - 4); // Remove trailing .WAV
    cpath += "_.wav";

    phones = std::move(tempPhones);
    loadNextClip(cpath, clip, -1);

    return CLIP_GOOD;
}

int Dataset::clipToFrames(const Clip& clip, size_t& nFrames, std::vector<Frame>& frames, ClassifierHelper& helper, const std::vector<Phone>& phones) {
    size_t fftStart = 0;
    bool hasInvalid = false;
    while (fftStart + FFT_REAL_SAMPLES < clip.size) {
        if (frames.size() <= nFrames) {
            frames.emplace_back();
            Frame& newFrame = frames.back();
            newFrame.reset();
        }
        Frame& currentFrame = frames[nFrames];
        helper.processFrame(clip.buffer.data(), fftStart, clip.size, frames, nFrames);
        if (Dataset::frameHasNan(currentFrame)) {
            currentFrame.invalid = true;
            hasInvalid = true;
            break;
        }

        size_t maxOverlap = 0;
        size_t maxIdx = 0;
        // Assign phoneme ID to frame
        currentFrame.phone = G_P_SIL; // Default silence
        for (int i = 0; i < phones.size(); i++) {
            const Phone& p = phones[i];

            if (p.maxIdx >= fftStart && fftStart >= p.minIdx) {
                currentFrame.phone = p.phonetic;

                break;
            }
        }

        fftStart += FFT_FRAME_SPACING;
        nFrames++;
    }
    if (hasInvalid)
        return CLIP_SKIP;
    
    return CLIP_GOOD;
}
