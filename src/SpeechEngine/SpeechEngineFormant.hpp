#pragma once

#include "SpeechEngine.hpp"

// Formant based speech synthesiser

class SpeechEngineFormant : public SpeechEngine {
public:
  SpeechEngineFormant() {};

  virtual void pushFrame(const SpeechFrame& frame);
  virtual void writeBuffer(OUTPUT_TYPE* outputBuffer, unsigned int nFrames);

  virtual SpeechEngineFormant& configure(std::string file);
protected:
  struct Formant {
    int freqIdx; // Equal to floor((frequency / freqStep) + 0.5)
    float volume;
    int width;
  };
  struct FormantGroup {
    std::vector<Formant> formants;
    float volume;
    size_t start;
    size_t end;
  };

  virtual void _init();

  size_t currentPhoneme = 0;
  size_t numFrequencies;
  const size_t freqStep = 20; // Inversely related to resolution of spectrum
  OUTPUT_TYPE** sineTables;
  size_t* tablePointers;
  size_t* tablePeriods;
  std::vector<FormantGroup> formantGroups;
  std::vector<FormantGroup> formantDatabank;
  size_t totalWrites = 0;

  // Reused arrays
  float* _spectrum;
  OUTPUT_TYPE* _tempBuffer;
};
