#pragma once

#include "../common_inc.hpp"

#include <string>
#include <vector>
#include <map>
#include "../structs.hpp"
#include "../Utils/Config.hpp"

class Voicebank {
public:
  struct UTAULine {
    std::string file;
    std::string alias;
    float offset;
    float consonant;
    float cutoff;
    float preutterance;
    float overlap;
  };
  struct Features {
    size_t from;
    std::vector<size_t> glide;
    size_t to;
  };
  struct Unit {
    // From oto
    std::string alias;
    uint32_t consonant; // Section to not loop = [0, consonant]; measured in number of samples since start
    uint32_t preutterance; // Starting point of the note; measured in number of samples since start
    uint32_t overlap; // Section to crossfade = [0, overlap]; measured in number of samples since start
    // Metadata about unit
    size_t index; // Index in voicebank's unit list
    Features features; // Parsed information about phonemes
    std::vector<float> audio;
    bool loaded; // True if audio is loaded into memory

    void save(int sr, const std::string& cacheDir) const;
    void unload();
    void load(const std::string& cacheDir);
  };
  struct DesiredFeatures {
    const Unit* prev;
    std::vector<size_t> glide;
    size_t to;
  };

  Voicebank() {};
  Voicebank& targetSamplerate(int sr);
  Voicebank& setShort(const std::string& name);
  Voicebank& open(const std::string& directory);

  const Unit& selectUnit(const DesiredFeatures& features);
  void loadUnit(size_t index);
  void unloadUnit(size_t index);
private:
  int unitCost(const DesiredFeatures& features, size_t index);
  void loadUnits(const std::vector<UTAULine>& lines);
  std::string bankName();
  std::string cacheDir();
  bool isCached();
  void saveCache();
  bool loadCache();
  static float cost(const Features& f1, const Features& f2);
  void loadAliases();

  Config config;
  std::string directory;
  std::string cacheDirectory;
  std::string shortName = "";
  int samplerate;
  std::vector<Unit> units;
  std::map<std::string, Features> aliasMapping;
  // Vector of list of indices
  // If phone "a" = 0, the value at index 0 contains a list of indices of units that end with phoneme "a"
  // In other words, units[index[0][0]] will get the first unit with an ending phoneme that has id 0
  std::vector<std::vector<size_t>> unitsIndex;
};
