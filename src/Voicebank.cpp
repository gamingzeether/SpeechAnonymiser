#include "Voicebank.h"

#include <filesystem>
#include <fstream>
#include <ctype.h>
#include <format>
#include <dr_wav.h>
#include <samplerate.h>

#define CACHE_VERSION 0

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
        saveCache();
    }

    return *this;
}

const Voicebank::Unit& Voicebank::selectUnit(const DesiredFeatures& features) {
    return units[0];
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
                wprintf(L"Failed to read voicebank wav: %s\n", filepath.c_str());
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
        size_t segmentLength = endSample - startSample;
        u.audio = std::vector<float>(segmentLength);
        memcpy(u.audio.data(), &resampledAudio[startSample], sizeof(float) * segmentLength);

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
    return std::filesystem::exists(cacheDir() + "/bank.json");
}

void Voicebank::saveCache() {
    if (!std::filesystem::exists(cacheDirectory + "/data")) {
        std::filesystem::create_directories(cacheDirectory + "/data");
    }
    json.open(cacheDirectory + "/bank.json", CACHE_VERSION);
    json["sample_rate"] = samplerate;

    JSONHelper::JSONObj jsonUnits = json.getRoot().add_arr("units");
    for (Unit& unit : units) {
        //unit.save(samplerate, cacheDirectory);
        JSONHelper::JSONObj jsonUnit = jsonUnits.append();
        // Need something in here so it knows how many units are in a bank
        jsonUnit["index"] = (int)unit.index;
        unit.unload();
    }
    json.save();
}

bool Voicebank::loadCache() {
    if (!isCached()) {
        return false;
    }

    bool result = json.open(cacheDirectory + "/bank.json", CACHE_VERSION);
    if (!result || 
        json["sample_rate"].get_int() != samplerate) {
        return false;
    }

    JSONHelper::JSONObj jsonUnits = json["units"];
    size_t unitCount = jsonUnits.get_array_size();
    units.resize(unitCount);
    for (int i = 0; i < unitCount; i++) {
        JSONHelper::JSONObj jsonUnit = jsonUnits[i];
        int index = jsonUnit["index"].get_int();
        Unit& u = units[index];
        u.index = i;
        u.audio = std::vector<float>();
        u.loaded = false;
    }

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
    drwav_uint64 framesWritten = drwav_write_pcm_frames(&wav, audio.size(), audio.data());
    drwav_uninit(&wav);
}

void Voicebank::Unit::load(const std::string& cacheDir) {
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