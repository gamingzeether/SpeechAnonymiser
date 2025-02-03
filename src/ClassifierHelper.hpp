#pragma once

#include "common_inc.hpp"

#include <vector>
#include <string>
#include <unordered_map>
#include <fftw3.h>
#include "structs.hpp"

class ClassifierHelper {
public:
    void initalize(size_t sr);
    void processFrame(const float* audio, const size_t& start, const size_t& totalSize, std::vector<Frame>& allFrames, size_t currentFrame);

    template <typename MatType>
    bool writeInput(const std::vector<Frame>& frames, const size_t& lastWritten, MatType& data, size_t col) {
        for (size_t f = 0; f < FFT_FRAMES; f++) {
            const Frame& readFrame = frames[(frames.size() + lastWritten - f) % frames.size()];
            if (readFrame.invalid)
                return false;
        }
        const size_t width = FFT_FRAMES;
        const size_t height = FRAME_SIZE;
        const size_t channels = 3;
        auto colPtr = data.colptr(col);
        for (size_t f = 0; f < FFT_FRAMES; f++) {
            const Frame& readFrame = frames[(frames.size() + lastWritten - f) % frames.size()];
            for (size_t i = 0; i < FRAME_SIZE; i++) {
                size_t x = f;
                size_t y = i;
                float r, g, b;
                if (true) {
                    r = readFrame.avg[i];
                    g = readFrame.delta[i];
                    b = readFrame.accel[i];
                } else {
                    float col = readFrame.avg[i];
                    if (col > 0) {
                        r = col * 2;
                        g = col;
                        b = 0;
                    } else {
                        r = 0;
                        g = 0;
                        b = -col;
                    }
                }
                colPtr[y * (width * channels) + x * channels + 0] = r;
                colPtr[y * (width * channels) + x * channels + 1] = g;
                colPtr[y * (width * channels) + x * channels + 2] = b;
            }
        }
        return true;
    };

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
