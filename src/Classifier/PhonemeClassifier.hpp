#pragma once

#include "../common_inc.hpp"

#include "PhonemeModel.hpp"
#include "Train/TSVReader.hpp"
#include "../Utils/Config.hpp"
#include "../include_mlpack.hpp"
#include "../define.hpp"
#include "../structs.hpp"

class PhonemeClassifier
{
public:
    bool ready;

    void initalize(const size_t& sr);
    void train(const std::string& path, const size_t& examples, const size_t& epochs);
    size_t classify(const CUBE_TYPE& data);
    std::string getPhonemeString(const size_t& in);
    void printConfusionMatrix(const CPU_CUBE_TYPE& testData, const CPU_CUBE_TYPE& testLabel, const arma::urowvec& lengths);
    void tuneHyperparam(const std::string& path, int iterations, int mr);
    void evaluate(const std::string& path);
    static double accuracy(NETWORK_TYPE& network, const CPU_CUBE_TYPE& data, const CPU_CUBE_TYPE& labels, const arma::urowvec& lengths);

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
    arma::Row<MAT_TYPE::elem_type> weighClasses(const CPU_CUBE_TYPE& labels, const arma::urowvec& lengths);

    Config config;

  PhonemeModel model;
    size_t sampleRate;

    bool initalized = false;
};
