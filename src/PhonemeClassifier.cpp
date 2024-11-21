#include "PhonemeClassifier.h"

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

// Update this when adding/remove json elements
#define CURRENT_VERSION 3
// Update this when modifying classifier parameters
#define CLASSIFIER_VERSION -2

#include <filesystem>
#include "ModelSerializer.h"
#include "Dataset.h"
#include "ClassifierHelper.h"

using namespace mlpack;

void PhonemeClassifier::initalize(const size_t& sr) {
    logger = Logger();
    logger.addStream(Logger::Stream("classifier.log").
        outputTo(Logger::VERBOSE).
        outputTo(Logger::INFO).
        outputTo(Logger::WARNING).
        outputTo(Logger::ERR).
        outputTo(Logger::FATAL));
    logger.addStream(Logger::Stream(std::cout).
        outputTo(Logger::INFO).
        outputTo(Logger::WARNING).
        outputTo(Logger::ERR).
        outputTo(Logger::FATAL));

    // Check if already initalized
    assert(!initalized);

    initalized = true;

    sampleRate = sr;

    ClassifierHelper::instance().initalize(sr);

    outputSize = ClassifierHelper::instance().inversePhonemeSet.size();

    // Load JSON
    bool openedJson = json.open("classifier.json", CURRENT_VERSION);

    optimizer = ens::Adam(
        STEP_SIZE,  // Step size of the optimizer.
        0, // Batch size. Number of data points that are used in each iteration.
        0.9,        // Exponential decay rate for the first moment estimates.
        0.999, // Exponential decay rate for the weighted infinity norm estimates.
        1e-8,  // Value used to initialise the mean squared gradient parameter.
        0, // Max number of iterations.
        1e-8,           // Tolerance.
        true);

    bool loaded = false;
    std::vector<size_t> inputDimensions = { FRAME_SIZE * 2, FFT_FRAMES, 1 };
    bool metaMatch = (
        openedJson &&
        json["classifier_version"].get_int() == CLASSIFIER_VERSION &&
        json["input_features"].get_int() == inputSize &&
        json["output_features"].get_int() == outputSize &&
        json["sample_rate"].get_int() == sampleRate);

    if (metaMatch && ModelSerializer::load(&network)) {
        logger.log("Loaded model", Logger::INFO);
        loaded = true;
    }

    if (!loaded) {
        logger.log("Model not loaded", Logger::WARNING);
        json["classifier_version"] = CLASSIFIER_VERSION;
        json["input_features"] = (int)inputSize;
        json["output_features"] = (int)outputSize;
        json["sample_rate"] = (int)sampleRate;
        float dropoutRate = 0.3;
        float l2regulatization = 0.01;

        network.Add<LinearNoBiasType<MAT_TYPE, L2Regularizer>>(1024, L2Regularizer(l2regulatization));
        network.Add<GELUType<MAT_TYPE>>();
        network.Add<DropoutType<MAT_TYPE>>(dropoutRate);

        network.Add<LinearType<MAT_TYPE, L2Regularizer>>(1024, L2Regularizer(l2regulatization));
        network.Add<GELUType<MAT_TYPE>>();
        network.Add<DropoutType<MAT_TYPE>>(dropoutRate);

        network.Add<LinearType<MAT_TYPE, L2Regularizer>>(1024, L2Regularizer(l2regulatization));
        network.Add<LeakyReLUType<MAT_TYPE>>();
        network.Add<DropoutType<MAT_TYPE>>(dropoutRate);

        network.Add<LinearType<MAT_TYPE, L2Regularizer>>(768, L2Regularizer(l2regulatization));
        network.Add<LeakyReLUType<MAT_TYPE>>();
        network.Add<DropoutType<MAT_TYPE>>(dropoutRate);

        network.Add<LinearType<MAT_TYPE, L2Regularizer>>(768, L2Regularizer(l2regulatization));
        network.Add<LeakyReLUType<MAT_TYPE>>();
        network.Add<DropoutType<MAT_TYPE>>(dropoutRate);

        network.Add<LinearType<MAT_TYPE, L2Regularizer>>(outputSize, L2Regularizer(l2regulatization));
        network.Add<LogSoftMaxType<MAT_TYPE>>();
    }
    network.InputDimensions() = inputDimensions;
    optimizer.ResetPolicy() = false;

    logger.log(std::format("Model initalized with {} input features and {} output features", inputSize, outputSize), Logger::INFO);

    ready = loaded;
    logger.log("Classifier ready", Logger::INFO);
}

void PhonemeClassifier::train(const std::string& path, const size_t& examples, const size_t& epochs, const double& stepSize) {
    optimizer.BatchSize() = 512;
    optimizer.StepSize() = stepSize;
    optimizer.MaxIterations() = epochs * examples * outputSize;
    optimizer.ResetPolicy() = false;

    std::vector<Phone> phones;
    
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
        logger.log(std::format("Starting loop {}", loops++), Logger::VERBOSE);

        train.start(inputSize, outputSize, examples, true);

        if (trainThread.joinable()) {
            trainThread.join();
        }

        test.start(inputSize, outputSize, examples / 5);
        validate.start(inputSize, outputSize, examples / 4);

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

            network.Train(CNAME(trainData),
                CNAME(trainLabel),
                optimizer,
                ens::PrintLoss(),
                ens::ProgressBar(50),
                ens::EarlyStopAtMinLossType<MAT_TYPE>(
                    [&](const MAT_TYPE& /* param */)
                    {
                        double validationLoss = network.Evaluate(CNAME(validateData), CNAME(validateLabel));
                        logger.log(std::format("Validation loss: {}", validationLoss), Logger::INFO);
                        if (validationLoss < bestLoss && epoch > 0) {
                            bestLoss = validationLoss;
                            ModelSerializer::save(&network, 999);
                            json.save();
                        }
                        if (epoch++ % 5 == 0) {
                            printConfusionMatrix(CNAME(trainData), CNAME(trainLabel));
                            printConfusionMatrix(testData, testLabel);
                            network.SetNetworkMode(true);
                        }
                        return validationLoss;
                    }));

            optimizer.StepSize() *= 0.5;
            ModelSerializer::save(&network, loops);
            json.save();
            });
        
        // Wait to finish copying data into new mat
        while (!copyDone) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
    }
}

size_t PhonemeClassifier::classify(const MAT_TYPE& data) {
    MAT_TYPE results = MAT_TYPE();
    network.Predict(data, results);
    float max = results(0);
    size_t maxIdx = 0;
    for (size_t i = 0; i < outputSize; i++) {
        float val = results(i);
        if (val > max) {
            max = val;
            maxIdx = i;
        }
    }
    return maxIdx;
}

std::string PhonemeClassifier::getPhonemeString(const size_t& in) {
    return ClassifierHelper::instance().inversePhonemeSet[in];
};

void PhonemeClassifier::printConfusionMatrix(const CPU_MAT_TYPE& testData, const CPU_MAT_TYPE& testLabel) {
    network.SetNetworkMode(false);
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
        std::cout << std::setw(2) << ClassifierHelper::instance().inversePhonemeSet[i] << " ";
    }
    std::cout << std::endl;
    for (size_t i = 0; i < outputSize; i++) {
        std::cout << std::setw(2) << ClassifierHelper::instance().inversePhonemeSet[i] << " ";
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
