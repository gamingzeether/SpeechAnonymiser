#pragma once

#include "common_inc.h"

#include "include_mlpack.h"
#include "TSVReader.h"
#include "JSONHelper.h"
#include "Logger.h"
#include "structs.h"
#include "define.h"

class PhonemeClassifier
{
public:
    bool ready;

    void initalize(const size_t& sr);
    void train(const std::string& path, const size_t& examples, const size_t& epochs, const double& stepSize);
    size_t classify(const MAT_TYPE& data);
    std::string getPhonemeString(const size_t& in);
    void printConfusionMatrix(const CPU_MAT_TYPE& testData, const CPU_MAT_TYPE& testLabel);

    inline size_t getInputSize() { return inputSize; };
    inline size_t getOutputSize() { return outputSize; };
    inline size_t getSampleRate() { return sampleRate; };

    void destroy() {
        if (!initalized) {
            return;
        }
        json.close();
    };
private:
    JSONHelper json;

    Logger logger;

	mlpack::FFN<mlpack::NegativeLogLikelihoodType<MAT_TYPE>, mlpack::RandomInitialization, MAT_TYPE> network;
	ens::Adam optimizer;
    size_t inputSize = FRAME_SIZE * FFT_FRAMES * 2;
    size_t outputSize = 0;
    size_t sampleRate;

    bool initalized = false;
};
