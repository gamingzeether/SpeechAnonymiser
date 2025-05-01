#include "ClassifierHelper.hpp"

#include <math.h>
#include <mlpack/core.hpp>
#include "Global.hpp"

void ClassifierHelper::initalize(size_t sr) {
  window = new float[FFT_FRAME_SAMPLES];
  for (int i = 0; i < FFT_FRAME_SAMPLES; i++) {
    //window[i] = 1.0; // None
    window[i] = 0.5f * (1.0f - cos((6.2831853f * i) / FFT_FRAME_SAMPLES)); // Hann
    //window[i] = 0.5f * (1.0f - cos((6.2831853f * i) / FFT_FRAME_SAMPLES)) * pow(2.7182818f, (-5.0f * abs(FFT_FRAME_SAMPLES - 2.0f * i)) / FFT_FRAME_SAMPLES); // Hann - Poisson
    //window[i] = 0.355768f - 0.487396f * cosf((6.28318530f * i) / FFT_FRAME_SAMPLES) - 0.144232 * cosf((12.5663706f * i) / FFT_FRAME_SAMPLES) - 0.012604 * cosf((18.8495559f * i) / FFT_FRAME_SAMPLES); // Nuttall
    //window[i] = 0.3635819 - 0.4891775 * cosf((6.28318530f * i) / FFT_FRAME_SAMPLES) - 0.1365995 * cosf((12.5663706f * i) / FFT_FRAME_SAMPLES) - 0.0106411 * cosf((18.8495559f * i) / FFT_FRAME_SAMPLES); // Blackman - Nuttall
  }

#pragma region Mel filterbank
  // Map fft to mel spectrum by index
  melTransform = new float* [MEL_BINS];
  melStart = new short[MEL_BINS];
  melEnd = new short[MEL_BINS];
  double melMax = 2595.0 * log10(1.0 + (sr / 700.0));
  double fftStep = (double)sr / FFT_REAL_SAMPLES;

  for (int melIdx = 0; melIdx < MEL_BINS; melIdx++) {
    melTransform[melIdx] = new float[FFT_REAL_SAMPLES];
    melStart[melIdx] = -1;
    melEnd[melIdx] = FFT_REAL_SAMPLES;
    double melFrequency = ((double)melIdx / MEL_BINS + (1.0 / MEL_BINS)) * melMax;

    for (int fftIdx = 0; fftIdx < FFT_REAL_SAMPLES; fftIdx++) {
      double frequency = (double)(fftIdx * sr) / FFT_REAL_SAMPLES;
      double fftFrequency = 2595.0 * log10(1.0 + (frequency / 700.0));

      double distance = abs(melFrequency - fftFrequency);
      distance /= fftStep;
      double effectMultiplier = 1.0 - (distance * distance);

      if (effectMultiplier > 0 && melStart[melIdx] == -1) {
        melStart[melIdx] = fftIdx;
      } else if (effectMultiplier <= 0 && melStart[melIdx] != -1 && melEnd[melIdx] == FFT_REAL_SAMPLES) {
        melEnd[melIdx] = fftIdx;
      }

      effectMultiplier = std::max(0.0, effectMultiplier);
      melTransform[melIdx][fftIdx] = effectMultiplier;
    }
  }
  // Normalize
  for (int melIdx = 0; melIdx < MEL_BINS; melIdx++) {
    double sum = 0;
    for (int fftIdx = 0; fftIdx < FFT_REAL_SAMPLES; fftIdx++) {
      sum += melTransform[melIdx][fftIdx];
    }
    double factor = 1.0 / sum;
    for (int fftIdx = 0; fftIdx < FFT_REAL_SAMPLES; fftIdx++) {
      melTransform[melIdx][fftIdx] *= factor;
    }
  }
#pragma endregion

  auto fftwLock = Global::get().fftwLock();
  fftwIn = fftwf_alloc_real(FFT_FRAME_SAMPLES);
  fftwOut = fftwf_alloc_complex(FFT_REAL_SAMPLES);
  fftwPlan = fftwf_plan_dft_r2c_1d(FFT_FRAME_SAMPLES, fftwIn, fftwOut, FFTW_MEASURE);
  dctIn = fftwf_alloc_real(MEL_BINS);
  dctOut = fftwf_alloc_real(DCT_BINS);
  dctPlan = fftwf_plan_r2r_1d(MEL_BINS, dctIn, dctOut, FFTW_REDFT10, FFTW_MEASURE);
}

void ClassifierHelper::processFrame(const float* audio, const size_t& start, const size_t& totalSize, std::vector<Frame>& allFrames, size_t currentFrame) {
  Frame& frame = allFrames[currentFrame];
  Frame& prevFrame = (currentFrame > 0) ? allFrames[currentFrame - 1] : frame;
  for (size_t i = 0; i < FFT_FRAME_SAMPLES; i++) {
    size_t readLocation = (start + i) % totalSize;
    const float value = audio[readLocation] * gain;
    fftwIn[i] = value * window[i];
  }

  // Do FFT
  fftwf_execute(fftwPlan);
  for (size_t i = 0; i < FFT_REAL_SAMPLES; i++) {
    fftwf_complex& complex = fftwOut[i];
    fftAmplitudes[i] = (complex[0] * complex[0] + complex[1] * complex[1]);
  }

  // To mel spectrum
  for (size_t melIdx = 0; melIdx < MEL_BINS; melIdx++) {
    melFrequencies[melIdx] = 0;
    for (size_t fftIdx = melStart[melIdx]; fftIdx < melEnd[melIdx]; fftIdx++) {
      const float& effect = melTransform[melIdx][fftIdx];
      melFrequencies[melIdx] += effect * fftAmplitudes[fftIdx];
    }
  }

  // Do DCT of logs
  for (size_t i = 0; i < MEL_BINS; i++) {
    float val = melFrequencies[i];
    dctIn[i] = log10(std::max(val, 1e-24f));
  }
  fftwf_execute(dctPlan);
  for (size_t i = 0; i < FRAME_SIZE; i++) {
    frame.real[i] = std::tanh(dctOut[i] / 50.0); // Clamp dct between -1 and 1
    //frame.real[i] = melFrequencies[i];
  }

  frame.volume = dctOut[0];
  
  // Write data
  for (size_t i = 0; i < FRAME_SIZE; i++) {
    frame.avg[i] = frame.real[i];
    frame.delta[i] = frame.avg[i] - prevFrame.avg[i];
    frame.accel[i] = frame.delta[i] - prevFrame.delta[i];
  }
}

bool ClassifierHelper::writeInput(const std::vector<Frame>& frames, const size_t lastWritten, CPU_CUBE_TYPE& data, const size_t col, const size_t slice) const {
  for (size_t f = 0; f < FFT_FRAMES; f++) {
    const Frame& readFrame = frames[(frames.size() + lastWritten - f) % frames.size()];
    if (readFrame.invalid)
      return false;
  }
  const size_t width = FFT_FRAMES;
  const size_t height = FRAME_SIZE;
  const size_t channels = 3;
  CPU_CUBE_TYPE tempCube;
  size_t offset = slice * data.n_elem_slice + col * data.n_rows;
  mlpack::MakeAlias(tempCube, data, 
      FFT_FRAMES, FRAME_SIZE, 3, offset);
  for (size_t r = 0; r < FFT_FRAMES; r++) {
    const Frame& readFrame = frames[(frames.size() + lastWritten - r) % frames.size()];
    for (size_t c = 0; c < FRAME_SIZE; c++) {
      tempCube(r, c, 0) = readFrame.avg[c];
      tempCube(r, c, 1) = readFrame.delta[c];
      tempCube(r, c, 2) = readFrame.accel[c];
    }
  }
  return true;
}
