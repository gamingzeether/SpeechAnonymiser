#include "Dataset.hpp"

#if 1
#define DO_NAN_CHECK
#endif

#include <algorithm>
#include <iostream>
#include <assert.h>
#include <dr_wav.h>
#include <mlpack/core/math/shuffle_data.hpp>
#include "ClassifierHelper.hpp"
#include "Util.hpp"
#include "Global.hpp"

#define CLIP_DURATION 12 // Max clip Duration

void Dataset::get(OUT CPU_CUBE_TYPE& data, OUT CPU_CUBE_TYPE& labels, arma::urowvec& sequenceLengths) {
    if (!done())
        return;
    auto lock = sharedData.lock();
    size_t outputSize = sharedData.exampleData.size();
    size_t inputSize = sharedData.exampleData.n_rows;

    data = sharedData.exampleData;
    labels = sharedData.exampleLabel;
    sequenceLengths = sharedData.sequenceLengths;
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

std::vector<float> Dataset::_findAndLoad(const std::string& path, size_t target, int samplerate, TSVReader::TSVLine& tsv, std::vector<Phone>& phones, const std::string& filter) {
    auto lock = sharedData.lock();
    sharedData.reader.shuffle();
    const std::string clipPath = path + "/clips/";
    const std::string transcriptsPath = path + "/transcript/";
    Clip clip;
    clip.type = COMMON_VOICE;
    clip.init(samplerate, CLIP_DURATION);
    std::vector<float> audio = std::vector<float>(samplerate * CLIP_DURATION);
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

        if (clipTooLong(phones))
            continue;

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
        loadNextClip(clipPath, tsv, clip, -1);
        clip.load(samplerate);
        for (size_t i = 0; i < clip.size; i++) {
            audio[i] = clip.buffer[i];
        }
        return audio;
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
    std::string path, clipPath;
    {
        auto lock = data.lock();
        type = data.type;
        sampleRate = data.sampleRate;
        clip.init(data.sampleRate, CLIP_DURATION);
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
    while (Dataset::keepLoading(data, _end)) {
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
                if (Dataset::clipTooLong(tempPhones))
                    continue;

                // Check if the clip should be loaded
                data.testedClips++;
                phones = std::move(tempPhones);
                loadNextClip(clipPath, nextClip, clip, -1);
            } else if (type == TIMIT) {
                if (!data.timitIter.good())
                    break;
                auto path = data.timitIter.next();
                auto fname = path.filename();
                std::vector<Phone> tempPhones = Dataset::parseTIMIT(path.string(), sampleRate);
                if (Dataset::clipTooLong(tempPhones))
                    continue;
                data.testedClips++;
                std::string cpath = path.string();
                cpath = cpath.substr(0, cpath.size() - 4);
                cpath += "_.wav";

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
            // Load all frames
            {
                size_t fftStart = 0;
                bool hasInvalid = false;
                while (fftStart + FFT_REAL_SAMPLES < clip.size) {
                    if (frames.size() <= frameCounter) {
                        frames.emplace_back();
                        Frame& newFrame = frames.back();
                        newFrame.reset();
                    }
                    Frame& currentFrame = frames[frameCounter];
                    helper.processFrame(clip.buffer.data(), fftStart, clip.size, frames, frameCounter);
                    if (Dataset::frameHasNan(currentFrame)) {
                        currentFrame.invalid = true;
                        hasInvalid = true;
                        break;
                    }

                    size_t maxOverlap = 0;
                    size_t maxIdx = 0;
                    // Assign phoneme ID to frame
                    currentFrame.phone = Global::get().silencePhone(); // Default silence
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
                if (hasInvalid)
                    continue;
            }

            // Find correct starts and ends of phones
            /*
            {
                std::vector<std::tuple<size_t, size_t>> ranges;
                std::vector<float> volumes;
                size_t lastPhone = frames[0].phone;
                size_t rangeStart = 0;
                // Gather information about the frames
                for (size_t i = 0; i < frames.size(); i++) {
                    const Frame& f = frames[i];
                    if (f.phone != lastPhone || i == frames.size() - 1) {
                        ranges.emplace_back(rangeStart, i - 1);
                        rangeStart = i;
                        lastPhone = f.phone;
                    }
                    volumes.push_back(f.volume);
                }
                // Can use this to find how load or quiet the clip is overall
                std::sort(volumes.begin(), volumes.end());
                float range = volumes.back() - volumes[0];
                float silentVolume = volumes[volumes.size() / 4];
                // Iterate over the frames in each phone
                for (const auto& range : ranges) {
                    size_t start = std::get<0>(range);
                    size_t end = std::get<1>(range);
                    for (size_t i = start; i <= end; i++) {
                        Frame& frame = frames[i];
                        if (frame.volume < silentVolume) {
                            // Treat it as silence
                            frame.phone = G_PS.fromString("");
                        }
                    }
                }
            }
            */

            // Write data into matrix
            {
                auto lock = data.lock();
                size_t writeCol = data.totalClips;
                if (writeCol >= data.exampleData.n_cols)
                    writeCol = rand() % data.exampleData.n_cols;
                size_t nSlices = 0;
                for (size_t i = FFT_FRAMES; i < frameCounter; i++) {
                    Frame& frame = frames[i - CONTEXT_BACKWARD];

                    // Write data
                    helper.writeInput(frames, i, data.exampleData, writeCol, nSlices);
                    data.exampleLabel(0, writeCol, nSlices) = frame.phone;
                    nSlices++;
                }
                data.sequenceLengths(writeCol) = nSlices;
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
