#pragma once

#include "common_inc.h"

#include "include_mlpack.h"
#include "TSVReader.h"
#include "Logger.h"
#include "PhonemeModel.h"
#include "structs.h"
#include "define.h"
#include "Config.h"

class PhonemeClassifier
{
public:
    bool ready;

    void initalize(const size_t& sr);
    void train(const std::string& path, const size_t& examples, const size_t& epochs);
    size_t classify(const MAT_TYPE& data);
    std::string getPhonemeString(const size_t& in);
    void printConfusionMatrix(const CPU_MAT_TYPE& testData, const CPU_MAT_TYPE& testLabel);
    void tuneHyperparam(const std::string& path, int iterations);

    inline size_t getInputSize() { return model.getInputSize(); };
    inline size_t getOutputSize() { return model.getOutputSize(); };
    inline size_t getSampleRate() { return model.getSampleRate(); };

    void destroy() {
        if (!initalized) {
            return;
        }
        config.close();
    };
private:
    Config config;

    Logger logger;

	PhonemeModel model;
    size_t sampleRate;

    bool initalized = false;
};
