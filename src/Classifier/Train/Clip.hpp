#pragma once

#include "../../common_inc.hpp"

#include <vector>
#include <string>
#include "DatasetTypes.hpp"

class Clip {
public:
  std::vector<float> buffer;
  float length;
  size_t size;
  unsigned int sampleRate;
  bool loaded = false;
  Type type;

  void setClipPath(const std::string& path);
  void load(int targetSampleRate);
  void init(size_t sr, float l) {
    length = l;
    buffer = std::vector<float>(sr * length);
  };

  Clip() {
    clipPath = "";
    size = 0;
    sampleRate = 0;
  }
private:
  float* loadMP3(OUT size_t& samples, OUT size_t& sampleRate, const std::string& path);
  float* loadWAV(OUT size_t& samples, OUT size_t& sampleRate, const std::string& path);
  void convertMono(float* buffer, OUT size_t& length, int channels);
  std::string clipPath;
};
