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
    void processFrame(Frame& frame, const float* audio, const size_t& start, const size_t& totalSize, const Frame& prevFrame);
    static size_t customHasher(const std::wstring& str);

    template <typename MatType>
    void writeInput(const std::vector<Frame>& frames, const size_t& lastWritten, MatType& data, size_t col) {
        for (size_t f = 0; f < FFT_FRAMES; f++) {
            const Frame& readFrame = frames[(lastWritten + frames.size() - f) % frames.size()];
            size_t offset = f * FRAME_SIZE;
            for (size_t i = 0; i < FRAME_SIZE; i++) {
                data(offset + i, col) = readFrame.real[i];
                //data(offset + i * 2 + 1, col) = readFrame.delta[i];
            }
        }
    };

    inline void setGain(float g) { gain = g; };

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

    void initalizePhonemeSet();

    ClassifierHelper() {};
};
