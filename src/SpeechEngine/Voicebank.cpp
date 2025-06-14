#include "Voicebank.hpp"

#include <filesystem>
#include <fstream>
#include <ctype.h>
#include <dr_wav.h>
#include <samplerate.h>
#include "../Utils/ClassifierHelper.hpp"
#include "../Utils/Logger.hpp"
#include "../Utils/Global.hpp"
#include "../Utils/Util.hpp"

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
      G_LG(Util::format("directory %s does not exist", directory.c_str()), Logger::DEAD);
    }
    std::ifstream iniReader;
    // Look for oto.ini
    {
      auto iterator = std::filesystem::directory_iterator(directory);
      std::string otoPath;
      // Case insensitive search for oto.ini
      for (const auto& entry : iterator) {
        std::string name = entry.path().filename().string();
        if (name.length() != std::string("oto.ini").length()) {
          continue;
        }
        for (int i = 0; i < name.length(); i++) {
          if (std::tolower(name[i]) != "oto.ini"[i]) {
            break;
          }
          iniReader = std::ifstream(entry.path());
        }
      }
      if (!iniReader.is_open()) {
        G_LG(Util::format("Could not find oto.ini in %s", directory.c_str()), Logger::DEAD);
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

    if (!std::filesystem::exists(cacheDirectory + "/data")) {
      std::filesystem::create_directories(cacheDirectory + "/data");
    }

    // Initalize aliases mapping
    if (std::filesystem::exists(cacheDirectory + "/aliases.json")) {
      std::filesystem::remove(cacheDirectory + "/aliases.json");
    }
    {
      // If the alias follows a specific format, translate it using a dictionary
      // Maps a sequence of characters to a phoneme used by ClassifierHelper
      std::map<std::string, std::string> charMapping;
      JSONHelper dictionaryConfig;
      if (dictionaryConfig.open("configs/alias_dictionary.json", 0)) {
        JSONHelper::JSONObj dictionary = dictionaryConfig["dictionary"];
        size_t dictSize = dictionary.get_array_size();
        for (size_t i = 0; i < dictSize; i++) {
          JSONHelper::JSONObj dictItem = dictionary[i];
          std::string phoneme = dictItem["phoneme"].get_string();
          charMapping[dictItem["sequence"].get_string()] = phoneme;
        }

        dictionaryConfig.close();
      } else {
        G_LG("Could not open alias dictionary", Logger::ERRO);
      }
      // Initalize alias file
      JSONHelper aliasConfig;
      aliasConfig.open(cacheDirectory + "/aliases.json", CACHE_VERSION);
      JSONHelper::JSONObj root = aliasConfig.getRoot();
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
              G_LG(Util::format("Could not find match: %s, %zd\n", aliasTmp.c_str(), pointer), Logger::WARN);
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
  
  // Initalize unitsIndex
  unitsIndex.resize(G_PS_S.size());
  for (Unit& u : units) {
    u.features = aliasMapping[u.alias];
    unitsIndex[u.features.to].push_back(u.index);
  }

  return *this;
}

Voicebank& Voicebank::setShort(const std::string& name) {
  shortName = name;
  return *this;
}

const Voicebank::Unit& Voicebank::selectUnit(const DesiredFeatures& features) {
  assert(features.to < unitsIndex.size());

  const auto& canidates = unitsIndex[features.to];
  if (canidates.size() == 0)
    G_LG(Util::format("No units with end %s", G_PS_S.xSampa(features.to).c_str()), Logger::WARN);

  size_t startPhoneme = (features.prev) ?
      features.prev->features.to :
      G_PS_S.xSampaIndex(".");

  int bestUnitCost = 9999;
  size_t unitIndex = -1;
  for (size_t i = 0; i < canidates.size(); i++) {
    size_t canidateIndex = canidates[i];
    int cost = unitCost(features, canidateIndex);
    if (cost < bestUnitCost) {
      bestUnitCost = cost;
      unitIndex = canidateIndex;
    }
  }

  return units[unitIndex];
}

int Voicebank::unitCost(const DesiredFeatures& features, size_t index) {
  const int WRONG_START_COST = 5;
  const int WRONG_END_COST = 10;
  const int PER_GLIDE_COST = 1;

  const Unit& indexUnit = units[index];
  size_t requestStartPhoneme = (features.prev) ? features.prev->features.to : G_P_SIL_S;

  int totalCost = 0;
  // Start phoneme mismatch cost
  totalCost += (indexUnit.features.from != requestStartPhoneme) ? WRONG_START_COST : 0;
  // End phoneme mismatch cost
  totalCost += (indexUnit.features.to != features.to) ? WRONG_END_COST : 0;
  // Glide phonemes cost
  totalCost += indexUnit.features.glide.size() * PER_GLIDE_COST;
  return totalCost;
}

void Voicebank::loadUnit(size_t index) {
  units[index].load(cacheDirectory);
}

void Voicebank::unloadUnit(size_t index) {
  units[index].unload();
}

void Voicebank::loadUnits(const std::vector<UTAULine>& lines) {
  std::string filename = "";

  std::vector<float> resampledAudio;
  size_t audioSamples;
  for (size_t i = 0; i < lines.size(); i++) {
    const UTAULine& line = lines[i];
    std::printf("Loading line %zd\r", i);

    if (line.file != filename) {
      // Done using previous audio file, load new one
      filename = line.file;
      std::string filepath = directory + filename;

      unsigned int chOut, srOut;
      drwav_uint64 samples;
      float* audio;
      audio = drwav_open_file_and_read_pcm_frames_f32(filepath.c_str(), &chOut, &srOut, &samples, NULL);
      if (audio == NULL || chOut <= 0 || srOut <= 0) {
        G_LG(Util::format("Failed to read voicebank wav: %s\n", filepath.c_str()), Logger::ERRO);
        continue;
      }

      // Reallocate if too small
      double ratio = (double)samplerate / srOut;
      drwav_uint64 outSize = samples * ratio;
      if (outSize > resampledAudio.size()) {
        resampledAudio.resize(outSize);
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
        memcpy(resampledAudio.data(), audio, sizeof(float) * samples);
        audioSamples = samples;
      } else {
        // Resample to target samplerate
        SRC_DATA upsampleData = SRC_DATA();
        upsampleData.data_in = audio;
        upsampleData.input_frames = samples;
        upsampleData.src_ratio = ratio;
        upsampleData.data_out = resampledAudio.data();
        upsampleData.output_frames = outSize;
        int error = src_simple(&upsampleData, SRC_SINC_BEST_QUALITY, 1);
        audioSamples = outSize;

        if (error) {
          G_LG(Util::format("Error while upsampling: %s", src_strerror(error)), Logger::ERRO);
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
    // Offset = left blank in milliseconds since start of file
    // Cutoff = right blank in milliseconds from end of file
    // If cutoff is negative, it is negative time from left blank
    size_t startSample = ((line.offset) / 1000) * samplerate;
    size_t endSample = (line.cutoff > 0) ?
        audioSamples - (line.cutoff / 1000) * samplerate :
        ((line.offset - line.cutoff) / 1000) * samplerate;
    endSample = std::min(endSample, audioSamples);
    size_t segmentLength = endSample - startSample;
    u.audio = std::vector<float>(segmentLength);
    for (size_t i = 0; i < segmentLength; i++) {
      u.audio[i] = resampledAudio[startSample + i];
    }

    // Copy info to unit
    u.alias = line.alias;
    u.consonant = (line.consonant / 1000) * samplerate;
    u.preutterance = (line.preutterance / 1000) * samplerate;
    u.overlap = (line.overlap / 1000) * samplerate;

    units.push_back(std::move(u));
  }
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

    u.alias = jsonUnit["alias"].get_string();
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
  drwav_init_file_write(&wav, (Util::format("%s/data/%ld.wav", cacheDir.c_str(), index)).c_str(), &format, NULL);
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
  std::string filepath = Util::format("%s/data/%ld.wav", cacheDir.c_str(), index);
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

//#define GEN_PS
void Voicebank::loadAliases() {
#ifdef GEN_PS
  JSONHelper newPhonemeSet;
  std::string psFile = "phoneme-set-" + shortName + ".json";
  if (std::filesystem::exists(psFile))
    std::filesystem::remove(psFile);
  newPhonemeSet.open(psFile);
  newPhonemeSet["name"] = shortName;
  JSONHelper::JSONObj mappings = newPhonemeSet.getRoot().add_arr("mappings");
  std::vector<std::string> addedPhones;
#endif

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
      size_t phonemeId = G_PS_S.xSampaIndex(xsampa);
      if (j == 0) {
        aliasFeatures.from = phonemeId;
      } else if (j == pcount - 1) {
        aliasFeatures.to = phonemeId;
      } else {
        aliasFeatures.glide.push_back(phonemeId);
      }

#ifdef GEN_PS
      if (!Util::contains(addedPhones, xsampa)) {
        JSONHelper::JSONObj map = mappings.append();
        map["token"] = xsampa;
        map["x_sampa"]["symbol"] = xsampa;
        addedPhones.push_back(xsampa);
      }
#endif
    }
    aliasMapping[aliasJson["name"].get_string()] = std::move(aliasFeatures);
  }

#ifdef GEN_PS
  newPhonemeSet.save();
#endif
}
