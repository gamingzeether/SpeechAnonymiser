﻿#include "PhonemeClassifier.h"

#ifdef __GNUC__
#define TYPE1 long long unsigned int
#define TYPE2 long unsigned int
#else
#define TYPE1 size_t
#define TYPE2 size_t 
#endif

#ifdef USE_GPU
#define CNAME(mat) gpu##mat
#define CONVERT(mat) MAT_TYPE CNAME(mat) = coot::conv_to<MAT_TYPE>::from(mat);
#else
#define CNAME(mat) mat
#endif

#include <filesystem>
#include "ModelSerializer.h"
#include "Dataset.h"
#include "ClassifierHelper.h"
#include "Global.h"

using namespace mlpack;

void PhonemeClassifier::initalize(const size_t& sr) {
    logger = Logger();
    logger.addStream(Logger::Stream("classifier.log")
        .outputTo(Logger::VERBOSE)
        .outputTo(Logger::INFO)
        .outputTo(Logger::WARNING)
        .outputTo(Logger::ERR)
        .outputTo(Logger::FATAL));
    logger.addStream(Logger::Stream(std::cout)
        .outputTo(Logger::INFO)
        .outputTo(Logger::WARNING)
        .outputTo(Logger::ERR)
        .outputTo(Logger::FATAL)
        .enableColor(true));

    // Check if already initalized
    assert(!initalized);

    initalized = true;

    sampleRate = sr;

    ClassifierHelper::instance().initalize(sr);

    PhonemeModel::Hyperparameters hp = PhonemeModel::Hyperparameters();
    hp.dropout() = 0.1;
    hp.l2() = 0.0001;
    hp.batchSize() = 128;
    hp.stepSize() = 0.001;
    model.setHyperparameters(hp);
    model.useLogger(logger);

    model.getSampleRate() = sampleRate;

    if (!model.load()) {
        logger.log("Model not loaded", Logger::WARNING, Logger::YELLOW);
        model.initModel();
    }
    model.initOptimizer();

    logger.log(std::format("Model initalized with {} input features and {} output features", model.getInputSize(), model.getOutputSize()), Logger::INFO);

    ready = true;
    logger.log("Classifier ready", Logger::INFO, Logger::GREEN);
}

void PhonemeClassifier::train(const std::string& path, const size_t& examples, const size_t& epochs) {
    //tuneHyperparam(path, 100);

    model.optimizer().MaxIterations() = epochs * examples * model.getOutputSize();

    std::vector<Phone> phones;
    
    int inputSize = model.getInputSize();
    int outputSize = model.getOutputSize();
    size_t* phonemeTracker = new size_t[outputSize];
    for (size_t i = 0; i < outputSize; i++) {
        phonemeTracker[i] = 1;
    }

    std::thread trainThread;
    bool isTraining = false;
    Dataset train(path + "/train.tsv", sampleRate, path);
    Dataset validate(path + "/dev.tsv", sampleRate, path);
    Dataset test(path + "/test.tsv", sampleRate, path);
    size_t loops = 0;
    double bestLoss = 9e+99;

    while (true) {
        train.start(inputSize, outputSize, examples, true);
        test.start(inputSize, outputSize, examples / 5);
        validate.start(inputSize, outputSize, examples / 4);

        if (trainThread.joinable()) {
            trainThread.join();
        }

        // Start training thread
        bool copyDone = false;
        trainThread = std::thread([&]{
            train.join();
            test.join();
            validate.join();

            CPU_MAT_TYPE testData, testLabel;
            test.get(testData, testLabel);
            CPU_MAT_TYPE trainData, trainLabel;
            train.get(trainData, trainLabel);
            CPU_MAT_TYPE validateData, validateLabel;
            validate.get(validateData, validateLabel);
            copyDone = true;
            int epoch = 0;

#ifdef USE_GPU
            CONVERT(trainData);
            CONVERT(trainLabel);
            CONVERT(validateData);
            CONVERT(validateLabel);
#endif

            model.optimizer().StepSize() = model.rate(1);
            logger.log(std::format("Starting training loop {}", loops++), Logger::INFO);
            model.network().Train(CNAME(trainData),
                CNAME(trainLabel),
                model.optimizer(),
                ens::PrintLoss(),
                ens::ProgressBar(50),
                ens::EarlyStopAtMinLossType<MAT_TYPE>(
                    [&](const MAT_TYPE& /* param */)
                    {
                        logger.log(std::format("Finished epoch {} with learning rate {}", epoch, model.optimizer().StepSize()), Logger::VERBOSE);
                        // Compare training and test accuracy to check for overfitting
                        if (epoch++ % 10 == 0) {
                            printConfusionMatrix(trainData, trainLabel);
                            printConfusionMatrix(testData, testLabel);
                            model.network().SetNetworkMode(true);
                        }

                        // Validation
                        double validationLoss = model.network().Evaluate(CNAME(validateData), CNAME(validateLabel));
                        logger.log(std::format("Validation loss: {}", validationLoss), Logger::INFO);
                        if (validationLoss < bestLoss && epoch > 0) {
                            bestLoss = validationLoss;
                            logger.log("Saving new best model", Logger::INFO);
                            model.save(999);
                        }
                        return validationLoss;
                    }, 20));

            model.save(loops);
            logger.log(std::format("Ended with best loss {}", bestLoss), Logger::INFO);
            });
        
        // Wait to finish copying data into new mat
        while (!copyDone) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
    }
}

size_t PhonemeClassifier::classify(const MAT_TYPE& data) {
    MAT_TYPE results = MAT_TYPE();
    model.network().Predict(data, results);
    float max = results(0);
    size_t maxIdx = 0;
    for (size_t i = 0; i < model.getOutputSize(); i++) {
        float val = results(i);
        if (val > max) {
            max = val;
            maxIdx = i;
        }
    }
    //std::printf("%.4f ", max);
    return maxIdx;
}

std::string PhonemeClassifier::getPhonemeString(const size_t& in) {
    return Global::get().phonemeSet().xSampa(in);
};

void PhonemeClassifier::printConfusionMatrix(const CPU_MAT_TYPE& testData, const CPU_MAT_TYPE& testLabel) {
    int inputSize = model.getInputSize();
    int outputSize = model.getOutputSize();

    model.network().SetNetworkMode(false);
    size_t testCount = testLabel.n_cols;
    size_t correctCount = 0;
    size_t* correctPhonemes = new size_t[outputSize];
    size_t** confusionMatrix = new size_t * [outputSize];
    size_t* totalPhonemes = new size_t[outputSize];
    for (size_t i = 0; i < outputSize; i++) {
        totalPhonemes[i] = 0;
    }
    for (size_t i = 0; i < outputSize; i++) {
        correctPhonemes[i] = 0;
        totalPhonemes[i] = 0;
        confusionMatrix[i] = new size_t[outputSize];
        for (size_t j = 0; j < outputSize; j++) {
            confusionMatrix[i][j] = 0;
        }
    }
    size_t testedExamples = 0;

    {
        for (size_t i = 0; i < testCount; i++) {
            MAT_TYPE input;
#ifdef USE_GPU
            input = coot::conv_to<MAT_TYPE>::from(testData.submat(arma::span(0, inputSize - 1), arma::span(i, i)));
#else
            input = testData.submat(arma::span(0, inputSize - 1), arma::span(i, i));
#endif
            size_t result = classify(input);
            size_t label = testLabel(0, i);
            if (result == label) {
                correctPhonemes[label]++;
                correctCount++;
            }
            confusionMatrix[label][result]++;
            totalPhonemes[label]++;
            testedExamples++;
        }
    }
    logger.log(std::format("Accuracy: {} out of {} ({:.1f}%)", (int)correctCount, (int)testedExamples, ((double)correctCount / testedExamples) * 100), Logger::INFO);
    std::cout << "Confusion Matrix:\n";
    std::cout << "   ";
    for (size_t i = 0; i < outputSize; i++) {
        std::cout << std::setw(2) << Global::get().phonemeSet().xSampa(i) << " ";
    }
    std::cout << std::endl;
    for (size_t i = 0; i < outputSize; i++) {
        std::cout << std::setw(2) << Global::get().phonemeSet().xSampa(i) << " ";
        size_t total = 0;
        for (size_t j = 0; j < outputSize; j++) {
            total += confusionMatrix[i][j];
        }
        for (size_t j = 0; j < outputSize; j++) {
            double fraction = (double)confusionMatrix[i][j] / total;
            int percent = fraction * 100;

            const char* format = (i == j) ? (
                (percent == 100) ? "\033[32m%2d\033[0m " /* 100% accuracy: green */ :
                "\033[36m%2d\033[0m ") /* diagonal: cyan */ :
                (percent > 0) ? "\033[31m%2d\033[0m " /* >= 1% misclassify: red */ :
                "%2d " /* everything else: white */;

            std::printf(format, (percent % 100));
        }
        std::cout << "\n";
    }

    delete[] totalPhonemes;
    delete[] correctPhonemes;
    for (size_t i = 0; i < outputSize; i++) {
        delete[] confusionMatrix[i];
    }
    delete[] confusionMatrix;
}

void PhonemeClassifier::tuneHyperparam(const std::string& path, int iterations) {
    int inputSize = model.getInputSize();
    int outputSize = model.getOutputSize();

    // Get data
    Dataset train(path + "/train.tsv", 16000, path);
    Dataset validate(path + "/dev.tsv", 16000, path);
    size_t tuneSize = 250;
    train.start(inputSize, outputSize, tuneSize * 0.8, true);
    validate.start(inputSize, outputSize, tuneSize * 0.2, true);
    train.join();
    validate.join();
    CPU_MAT_TYPE tuneTrainData, tuneTrainLabel;
    train.get(tuneTrainData, tuneTrainLabel);
    CPU_MAT_TYPE tuneValidData, tuneValidLabel;
    validate.get(tuneValidData, tuneValidLabel);
#ifdef USE_GPU
    CONVERT(tuneTrainData);
    CONVERT(tuneTrainLabel);
    CONVERT(tuneValidData);
    CONVERT(tuneValidLabel);
#endif

    // Prepare models
    std::vector<PhonemeModel> models;
    std::vector<float> bestLosses;
    int mr = 2;
    for (int i = 0; i < 2 * mr + 1; i++) {
        PhonemeModel mdl;
        models.push_back(mdl);
        bestLosses.push_back(9e99);
    }
    std::vector<std::thread> trainThreads(models.size());

    // Initalize parameters
    PhonemeModel::Hyperparameters base = PhonemeModel::Hyperparameters();
    base.dropout() = 0.3;
    base.l2() = 0.001;
    base.batchSize() = 512;
    base.stepSize() = 0.005;
    base.warmup() = 10;

    PhonemeModel::Hyperparameters stepSize = PhonemeModel::Hyperparameters();
    stepSize.dropout() = 0.05;
    stepSize.l2() = 0.0001;
    stepSize.batchSize() = 256;
    stepSize.stepSize() = 0.0001;
    stepSize.warmup() = 1;

    int paramSize = PhonemeModel::Hyperparameters::size;

    int optimEpochs = 500;
    int optimIterations = optimEpochs * (tuneSize * 0.8) * outputSize;

    // Do iterations
    // Each iteration adjusts one parameter up or down various amounts per model
    // Then continues with the best out of them
    for (int i = 0; i < iterations * paramSize; i++) {
        int targetParam = i % paramSize;
        float delta = stepSize.e[targetParam];

        logger.log(std::format("Starting iteration {}, subiteration {}", i / paramSize, i % paramSize), Logger::INFO);
        for (int j = 0; j < models.size(); j++) {
            // Reset stuff
            bestLosses[j] = 9e99;

            // Initalize models with parameters
            PhonemeModel::Hyperparameters tempParams = base;
            tempParams.e[targetParam] += delta * (j - mr);

            PhonemeModel& mdl = models[j];
            mdl.setHyperparameters(tempParams);
            mdl.initModel();
            mdl.initOptimizer();
            // Copy first model's inital weights
            if (j > 0) {
                mdl.network().Parameters() = models[0].network().Parameters();
            }

            // Start training models
            logger.log(std::format("Starting model {}", j), Logger::INFO);
            std::thread& trainThread = trainThreads[j];
            trainThread = std::thread([this, &mdl, &CNAME(tuneTrainData), &CNAME(tuneTrainLabel), &CNAME(tuneValidData), &CNAME(tuneValidLabel), &bestLosses, j, optimIterations] {
                mdl.optimizer().MaxIterations() = optimIterations;

                mdl.network().Train(
                    CNAME(tuneTrainData),
                    CNAME(tuneTrainLabel),
                    mdl.optimizer(),
                    ens::EarlyStopAtMinLossType<MAT_TYPE>(
                        [&](const MAT_TYPE& /* param */)
                        {
                            double validationLoss = mdl.network().Evaluate(CNAME(tuneValidData), CNAME(tuneValidLabel));
                            if (validationLoss < bestLosses[j]) {
                                bestLosses[j] = validationLoss;
                                logger.log(std::format("Model {} with new best {}", j, validationLoss), Logger::INFO);
                            }
                            return validationLoss;
                        })
                );
                logger.log(std::format("Model {} done", j), Logger::INFO);
                });
        }

        // Wait for training to complete
        for (int j = 0; j < models.size(); j++) {
            if (trainThreads[j].joinable()) {
                trainThreads[j].join();
            }
        }

        // Evaluate best
        int bestIndex = -1;
        float bestLoss = 9e99;
        for (int j = 0; j < models.size(); j++) {
            float mLoss = bestLosses[j];
            if (mLoss < bestLoss) {
                bestLoss = mLoss;
                bestIndex = j;
            }
        }
        float bestDelta = delta * (bestIndex - mr);
        base.e[targetParam] += bestDelta;

        // Print new best
        logger.log(std::format("Best hyperparameters after iteration {}, subiteration {}", i / paramSize, i % paramSize), Logger::INFO);
        for (int j = 0; j < paramSize; j++) {
            logger.log(std::format("{}", base.e[j]), Logger::INFO);
        }
    }
}
