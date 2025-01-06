#pragma once

#include "common_inc.h"

#include <vector>
#include <string>
#include <unordered_map>
#include <fftw3.h>
#include "structs.h"

class ClassifierHelper {
public:
    static ClassifierHelper& instance() {
        static ClassifierHelper s;
        return s;
    };

    void initalize(size_t sr);
    void processFrame(const float* audio, const size_t& start, const size_t& totalSize, std::vector<Frame>& allFrames, size_t currentFrame);

    template <typename MatType>
    bool writeInput(const std::vector<Frame>& frames, const size_t& lastWritten, MatType& data, size_t col) {
        for (size_t f = 0; f < FFT_FRAMES; f++) {
            const Frame& readFrame = frames[(lastWritten - f) % frames.size()];
            if (readFrame.invalid)
                return false;
        }
        auto colPtr = data.colptr(col);
        for (size_t f = 0; f < FFT_FRAMES; f++) {
            const Frame& readFrame = frames[(lastWritten - f) % frames.size()];
            for (size_t i = 0; i < FRAME_SIZE; i++) {
                colPtr[(0 * (FFT_FRAMES * FRAME_SIZE)) + (i * FFT_FRAMES) + f] = readFrame.avg[i];
                colPtr[(1 * (FFT_FRAMES * FRAME_SIZE)) + (i * FFT_FRAMES) + f] = readFrame.delta[i];
                colPtr[(2 * (FFT_FRAMES * FRAME_SIZE)) + (i * FFT_FRAMES) + f] = readFrame.accel[i];
            }
        }
        return true;
    };

    inline void setGain(float g) { gain = g; };
private:
    float gain;
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

    ClassifierHelper() {};
};
