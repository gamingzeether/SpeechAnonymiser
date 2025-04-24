#pragma once

#include "../common_inc.hpp"

#include <chrono>
#include <random>
#include <unordered_map>
#include <string>
#include <mutex>
#include "../Utils/JSONHelper.hpp"
#include "../define.hpp"
#include "../structs.hpp"

/* Defines a base speech engine
 * Do not use this
 */

class SpeechEngine {
public:
  // Pass frame containing phoneme to engine
  virtual void pushFrame(const SpeechFrame& frame) = 0;
  // Write audio into specified buffer
  virtual void writeBuffer(OUTPUT_TYPE* outputBuffer, unsigned int nFrames) = 0;

  virtual SpeechEngine& setSampleRate(int sr) { 
    sampleRate = sr;
    return *this;
  }
  virtual SpeechEngine& setChannels(int ch) {
    channels = ch;
    return *this;
  }
  virtual SpeechEngine& setVolume(float v) {
    volume = v;
    return *this;
  }

  // Last step in setting up the speech engine
  // Configures and initalizes the engine for use
  virtual SpeechEngine& configure(std::string file) = 0;
protected:
  static float _random() {
    std::uniform_real_distribution<float> randomDist = std::uniform_real_distribution<float>(-1, 1);
    return randomDist(randomEngine);
  };
  virtual void _init() = 0;

  int sampleRate = 16000;
  int channels = 1;
  float volume = 1;
  std::mutex stateMutex;

  inline static std::default_random_engine randomEngine = std::default_random_engine((unsigned int)std::chrono::system_clock::now().time_since_epoch().count());
};
