#pragma once

#include "../common_inc.hpp"

#include <vector>
#include <string>
#include <unordered_map>
#include <armadillo>
#include <fftw3.h>
#include "../structs.hpp"

class ClassifierHelper {
public:
  void initalize(size_t sr);
  void processFrame(const float* audio, const size_t& start, const size_t& totalSize, std::vector<Frame>& allFrames, size_t currentFrame);
  bool writeInput(const std::vector<Frame>& frames, const size_t lastWritten, CPU_CUBE_TYPE& data, const size_t col, const size_t slice) const;

  inline void setGain(float g) { gain = g; };
private:
  float gain = 1;
  float* window;
  float* fftwIn;
  fftwf_complex* fftwOut;
  fftwf_plan fftwPlan;
  float* dctIn;
  float* dctOut;
  fftwf_plan dctPlan;

  float** melTransform;
  short* melStart;
  short* melEnd;

  float* fftAmplitudes = new float[FFT_REAL_SAMPLES];
  float* melFrequencies = new float[MEL_BINS];
  float* windowAvg = new float[FFT_REAL_SAMPLES];
};
