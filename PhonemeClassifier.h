#pragma once

#include "common_inc.h"

#include <fftw3.h>
#include <mlpack/mlpack.hpp>
#include "TSVReader.h"
#include "structs.h"
#include "define.h"

class PhonemeClassifier
{
public:
    struct Clip {
        std::string clipPath;
        std::string* tsvElements;
        float* buffer;
        size_t size;
        std::string sentence;
        unsigned int sampleRate;
        bool loaded = false;
        bool isTest = false;

        void loadMP3(int targetSampleRate);
        void initSampleRate(size_t sr) {
            buffer = new float[sr * CLIP_LENGTH];
        };

        Clip() {
            clipPath = "";
            tsvElements = NULL;
            size = 0;
            sentence = "";
            sampleRate = 0;
        }
        ~Clip() {
            delete[] tsvElements;
            delete[] buffer;
        }
    };

    bool ready;

    void initalize(const size_t& sr, bool load);
    void train(const std::string& path, const size_t& batchSize, const size_t& epochs);
    size_t classify(const arma::mat& data);
    void processFrame(Frame& frame, const float* audio, const size_t& start, const size_t& totalSize, const Frame& prevFrame);
    void writeInput(const std::vector<Frame>& frames, const size_t& lastWritten, arma::mat& data, size_t col);
    std::string getPhonemeString(const size_t& in) { return inversePhonemeSet[in]; };
    void preprocessDataset(const std::string& path);

    inline size_t getInputSize() { return inputSize; };
    inline size_t getOutputSize() { return outputSize; };
    inline size_t getSampleRate() { return SAMPLE_RATE; };
    inline void setGain(float g) { gain = g; };

    void destroy() {
        if (!initalized) {
            return;
        }
        fftwf_destroy_plan(fftwPlan);
        free(fftwIn);
        fftwf_free(fftwOut);
        delete[] window;
    };
private:
    std::unordered_map<size_t, size_t> phonemeSet;
    std::vector<std::string> inversePhonemeSet;

	mlpack::FFN<mlpack::NegativeLogLikelihood, mlpack::RandomInitialization> network;
	ens::Adam optimizer;
    size_t inputSize = FFT_REAL_SAMPLES * FFT_FRAMES;
    size_t outputSize = 0;
    size_t SAMPLE_RATE;

    bool initalized = false;
    float* window;
    float* fftwIn;
    fftwf_complex* fftwOut;
    fftwf_plan fftwPlan;
    float gain;

    float** melTransform;
    short* melStart;
    short* melEnd;

	// https://stackoverflow.com/a/7154226
	static std::wstring utf8_to_utf16(const std::string& utf8);
    static size_t customHasher(const std::wstring& str);
    void loadNextClip(const std::string& clipPath, TSVReader& tsv, OUT Clip& clip, int sampleRate);
    void initalizePhonemeSet();
};

