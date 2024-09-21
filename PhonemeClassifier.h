#pragma once

#include "common_inc.h"

#include <mlpack/mlpack.hpp>
#include "TSVReader.h"
#include "JSONHelper.h"
#include "structs.h"
#include "define.h"

class PhonemeClassifier
{
public:
    bool ready;

    void initalize(const size_t& sr);
    void train(const std::string& path, const size_t& examples, const size_t& epochs, const double& stepSize);
    size_t classify(const arma::mat& data);
    std::string getPhonemeString(const size_t& in);

    inline size_t getInputSize() { return inputSize; };
    inline size_t getOutputSize() { return outputSize; };
    inline size_t getSampleRate() { return SAMPLE_RATE; };

    void destroy() {
        if (!initalized) {
            return;
        }
        json.close();
    };
private:
    JSONHelper json;

	mlpack::FFN<mlpack::NegativeLogLikelihood, mlpack::RandomInitialization> network;
	ens::Adam optimizer;
    size_t inputSize = FRAME_SIZE * FFT_FRAMES;
    size_t outputSize = 0;
    size_t SAMPLE_RATE;

    bool initalized = false;
};
