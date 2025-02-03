#include "Voicebank.hpp"

#include <filesystem>
#include <fstream>
#include <ctype.h>
#include <format>
#include <dr_wav.h>
#include <samplerate.h>
#include "ClassifierHelper.hpp"
#include "Logger.hpp"
#include "Global.hpp"
#include "Util.hpp"

#define CACHE_VERSION 1
#define DICT_WIDTH 5

std::string split(std::string& in, char c) {
    std::string left = in.substr(0, in.find_first_of(c));
    in = in.substr(in.find_first_of(c) + 1);
    return left;
}

Voicebank& Voicebank::targetSamplerate(int sr) {
    samplerate = sr;
    return *this;
}

Voicebank& Voicebank::open(const std::string& dir) {
    directory = dir;
    cacheDirectory = cacheDir();

    config = Config(cacheDirectory + "/bank.json", CACHE_VERSION);

    if (!loadCache()) {
        // Open and read oto.ini into lines
        if (!std::filesystem::is_directory(directory)) {
            throw("directory %s does not exist\n", directory.c_str());
        }
        std::ifstream iniReader;
        // Look for oto.ini
        {
            auto iterator = std::filesystem::directory_iterator(directory);
            std::string otoPath;
            bool match = false;
            for (const auto& entry : iterator) {
                std::string name = entry.path().filename().string();
                if (name.length() != std::string("oto.ini").length()) {
                    continue;
                }
                for (int i = 0; i < name.length(); i++) {
                    if (std::tolower(name[i]) != "oto.ini"[i]) {
                        match = false;
                        break;
                    }
                    match = true;
                }
                if (match) {
                    iniReader = std::ifstream(entry.path());
                    break;
                }
            }
            if (!match) {
                throw("Could not find oto.ini in %s\n", directory.c_str());
            }
        }
        // Parse lines
        std::vector<UTAULine> lines;
        while (true) {
            std::string line;
            std::getline(iniReader, line);
            if (line == "") {
                break;
            }
            UTAULine uline;

            uline.file = split(line, '=');
            uline.alias = split(line, ',');
            uline.offset = std::stof(split(line, ','));
            uline.consonant = std::stof(split(line, ','));
            uline.cutoff = std::stof(split(line, ','));
            uline.preutterance = std::stof(split(line, ','));
            uline.overlap = std::stof(line);

            lines.push_back(std::move(uline));
        }

        loadUnits(lines);

        // Initalize aliases mapping
        if (std::filesystem::exists(cacheDirectory + "/aliases.json")) {
            std::filesystem::remove(cacheDirectory + "/aliases.json");
        }
        {
            // If the alias follows a specific format, translate it using a dictionary
            // Maps a sequence of characters to a phoneme used by ClassifierHelper
            std::map<std::string, std::string> charMapping;
            if (std::filesystem::exists("configs/alias_dictionary.json")) {
                Config dictionaryConfig = Config("configs/alias_dictionary.json", 0);
                dictionaryConfig.load();

                JSONHelper::JSONObj dictionary = dictionaryConfig.object()["dictionary"];
                size_t dictSize = dictionary.get_array_size();
                auto set = G_PS;
                for (size_t i = 0; i < dictSize; i++) {
                    JSONHelper::JSONObj dictItem = dictionary[i];
                    std::string phoneme = dictItem["phoneme"].get_string();
                    // Check to see if it is a valid phoneme
                    size_t pidx = G_PS.xSampaExists(phoneme);
                    charMapping[dictItem["sequence"].get_string()] = phoneme;
                }

                dictionaryConfig.close();
            }
            // Initalize alias file
            Config aliasConfig = Config(cacheDirectory + "/aliases.json", CACHE_VERSION);
            aliasConfig.load(); // This will initalize a new file
            JSONHelper::JSONObj root = aliasConfig.object().getRoot();
            JSONHelper::JSONObj arr = root.add_arr("aliases");
            // Write entries
            for (const UTAULine& line : lines) {
                JSONHelper::JSONObj alias = arr.append();
                alias["name"] = line.alias;
                JSONHelper::JSONObj list = alias.add_arr("phonemes");
                size_t pointer = 0;
                std::string aliasTmp = line.alias.substr(0, line.alias.length() - shortName.length());
                while (pointer < aliasTmp.size()) {
                    for (int i = DICT_WIDTH; i >= 0; i--) {
                        // Could not match
                        if (i == 0) {
                            std::printf("Could not find match: %s, %zd\n", aliasTmp.c_str(), pointer);
                            pointer++;
                            break;
                        }
                        std::string chk = aliasTmp.substr(pointer, i);
                        auto iter = charMapping.find(chk);
                        if (iter != charMapping.end()) {
                            // Found a match
                            JSONHelper::JSONObj item = list.append();
                            item = iter->second;
                            pointer += i;
                            break;
                        }
                    }
                }
            }
            aliasConfig.save();
        }

        saveCache();
    }

    loadAliases();

    return *this;
}

Voicebank& Voicebank::setShort(const std::string& name) {
    shortName = name;
    return *this;
}

const Voicebank::Unit& Voicebank::selectUnit(const DesiredFeatures& features) {
    assert(features.to < units.size());
    return units[features.to];
}

void Voicebank::loadUnit(size_t index) {
    units[index].load(cacheDirectory);
}

void Voicebank::unloadUnit(size_t index) {
    units[index].unload();
}

void Voicebank::loadUnits(const std::vector<UTAULine>& lines) {
    std::string filename = "";

    float* resampledAudio = NULL;
    size_t audioSamples;
    size_t allocatedLength = 0;
    for (size_t i = 0; i < lines.size(); i++) {
        const UTAULine& line = lines[i];
        std::printf("Loading line %zd\r", i);

        // Load new wav
        if (line.file != filename) {
            filename = line.file;
            std::string filepath = directory + filename;

            unsigned int chOut, srOut;
            drwav_uint64 samples;
            float* audio;
            audio = drwav_open_file_and_read_pcm_frames_f32(filepath.c_str(), &chOut, &srOut, &samples, NULL);
            if (audio == NULL || chOut <= 0 || srOut <= 0) {
                std::printf("Failed to read voicebank wav: %s\n", filepath.c_str());
                continue;
            }

            // Reallocate if too small
            double ratio = (double)samplerate / srOut;
            long outSize = (long)(ratio * samples);
            if (samples > allocatedLength) {
                delete[] resampledAudio;
                resampledAudio = new float[(size_t)outSize];
                allocatedLength = outSize;
            }

            // Single channel
            if (chOut > 1) {
                size_t realSamples = samples / chOut;
                for (int j = 0; j < realSamples; j++) {
                    float sum = 0;
                    size_t readIdx = j * chOut;
                    for (int k = 0; k < chOut; k++) {
                        sum += audio[readIdx + k];
                    }
                    audio[j] = sum / chOut;
                }
                samples = realSamples;
            }

            // Copy audio to unit
            if (srOut == samplerate) {
                // 1 to 1 copy
                memcpy(resampledAudio, audio, sizeof(float) * samples);
                audioSamples = samples;
            } else {
                // Resample to target samplerate
                SRC_DATA upsampleData = SRC_DATA();
                upsampleData.data_in = audio;
                upsampleData.input_frames = samples;
                upsampleData.src_ratio = ratio;
                upsampleData.data_out = resampledAudio;
                upsampleData.output_frames = outSize;
                int error = src_simple(&upsampleData, SRC_SINC_BEST_QUALITY, chOut);
                audioSamples = outSize;

                if (error) {
                    std::cout << "Error while upsampling: " << src_strerror(error) << '\n';
                }
            }
            free(audio);
        }

        // Initalize unit
        size_t index = units.size();
        Unit u = Unit();

        u.index = index;
        u.loaded = true;
        // Copy specified audio into unit
        if (line.cutoff >= 0) {
            continue;
        }
        size_t startSample = ((line.offset) / 1000) * samplerate;
        size_t endSample = ((line.offset - line.cutoff) / 1000) * samplerate;
        endSample = std::min(endSample, audioSamples);
        size_t segmentLength = endSample - startSample;
        u.audio = std::vector<float>(segmentLength);
        memcpy(u.audio.data(), &resampledAudio[startSample], sizeof(float) * segmentLength);

        // Copy info to unit
        u.alias = line.alias;
        u.consonant = (line.consonant / 1000) * samplerate;
        u.preutterance = (line.preutterance / 1000) * samplerate;
        u.overlap = (line.overlap / 1000) * samplerate;

        units.push_back(std::move(u));
    }
    delete[] resampledAudio;
}

std::string Voicebank::bankName() {
    std::string tmp = directory;
    if (tmp.back() == '/')
        tmp = tmp.substr(0, tmp.size() - 1);
    return tmp.substr(tmp.find_last_of('/') + 1);
}

std::string Voicebank::cacheDir() {
    return "cache/" + bankName();
}

bool Voicebank::isCached() {
    config.setDefault("sample_rate", samplerate);
    config.load();
    return config.matchesDefault();
}

void Voicebank::saveCache() {
    if (!std::filesystem::exists(cacheDirectory + "/data")) {
        std::filesystem::create_directories(cacheDirectory + "/data");
    }

    JSONHelper::JSONObj jsonUnits = config.object().getRoot().add_arr("units");
    for (Unit& unit : units) {
        unit.save(samplerate, cacheDirectory);
        JSONHelper::JSONObj jsonUnit = jsonUnits.append();
        // Need something in here so it knows how many units are in a bank
        jsonUnit["index"] = (int)unit.index;
        jsonUnit["alias"] = unit.alias;
        jsonUnit["consonant"] = (int)unit.consonant;
        jsonUnit["preutterance"] = (int)unit.preutterance;
        jsonUnit["overlap"] = (int)unit.overlap;
        unit.unload();
    }
    config.save();
}

bool Voicebank::loadCache() {
    if (!isCached()) {
        config.useDefault();
        return false;
    }

    JSONHelper::JSONObj jsonUnits = config.object()["units"];
    size_t unitCount = jsonUnits.get_array_size();
    units.resize(unitCount);
    for (size_t i = 0; i < unitCount; i++) {
        JSONHelper::JSONObj jsonUnit = jsonUnits[i];
        int index = jsonUnit["index"].get_int();
        Unit& u = units[index];
        u.index = i;
        u.audio = std::vector<float>();
        u.loaded = false;
        u.consonant = jsonUnit["consonant"].get_int();
        u.preutterance = jsonUnit["preutterance"].get_int();
        u.overlap = jsonUnit["overlap"].get_int();
    }

    loadAliases();

    return true;
}

void Voicebank::Unit::save(int sr, const std::string& cacheDir) const {
    drwav wav;
    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_PCM;
    format.channels = 1;
    format.sampleRate = sr;
    format.bitsPerSample = 16;
    drwav_init_file_write(&wav, (std::format("{}/data/{}.wav", cacheDir, index)).c_str(), &format, NULL);
    // Convert data to 16 bit signed int
    int16_t* intData = new int16_t[audio.size()];
    size_t samples = audio.size();
    for (int i = 0; i < samples; i++) {
        intData[i] = std::numeric_limits<int16_t>::max() * audio[i];
    }
    // Write data
    drwav_uint64 framesWritten = drwav_write_pcm_frames(&wav, samples, intData);
    // Cleanup
    delete[] intData;
    drwav_uninit(&wav);
}

void Voicebank::Unit::load(const std::string& cacheDir) {
    if (loaded)
        return;
    std::string filepath = std::format("{}/data/{}.wav", cacheDir, index);
    unsigned int chOut, srOut;
    drwav_uint64 samples;
    float* data;
    data = drwav_open_file_and_read_pcm_frames_f32(filepath.c_str(), &chOut, &srOut, &samples, NULL);
    audio.resize(samples);
    memcpy(audio.data(), data, sizeof(float) * samples);
    free(data);
    loaded = true;
}

void Voicebank::Unit::unload() {
    audio.clear();
    audio.shrink_to_fit();
    loaded = false;
}

void Voicebank::loadAliases() {
    Config aliasConfig = Config(cacheDirectory + "/aliases.json", CACHE_VERSION);
    aliasConfig.load();
    JSONHelper::JSONObj root = aliasConfig.object().getRoot();
    JSONHelper::JSONObj arr = root["aliases"];
    int count = root["aliases"].get_array_size();
    for (int i = 0; i < count; i++) {
        JSONHelper::JSONObj aliasJson = arr[i];
        Features aliasFeatures;
        aliasFeatures.glide = std::vector<size_t>();
        JSONHelper::JSONObj phonemes = aliasJson["phonemes"];
        int pcount = phonemes.get_array_size();
        for (int j = 0; j < pcount; j++) {
            std::string xsampa = phonemes[j].get_string();
            size_t phonemeId = G_PS.xSampaIndex(xsampa);
            if (j == 0) {
                aliasFeatures.from = phonemeId;
            } else if (j == pcount - 1) {
                aliasFeatures.to = phonemeId;
            } else {
                aliasFeatures.glide.push_back(phonemeId);
            }
        }
        aliasMapping[aliasJson["name"].get_string()] = std::move(aliasFeatures);
    }
}
