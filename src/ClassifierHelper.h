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
    static size_t customHasher(const std::wstring& str);

    template <typename MatType>
    void writeInput(const std::vector<Frame>& frames, const size_t& lastWritten, MatType& data, size_t col) {
        for (size_t f = 0; f < FFT_FRAMES; f++) {
            const Frame& readFrame = frames[(lastWritten + frames.size() - f) % frames.size()];
            size_t offset = f * FRAME_SIZE;
            for (size_t i = 0; i < FRAME_SIZE; i++) {
                data(offset + i, col) = readFrame.avg[i];
            }
        }
    };

    inline void setGain(float g) { gain = g; };
    // https://stackoverflow.com/a/7154226
    static std::wstring utf8_to_utf16(const std::string& utf8);

    std::unordered_map<size_t, size_t> phonemeSet;
    std::vector<std::string> inversePhonemeSet;
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

    void initalizePhonemeSet();

    ClassifierHelper() {};
};
