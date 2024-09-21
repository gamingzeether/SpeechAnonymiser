#include "PhonemeClassifier.h"

#ifdef __GNUC__
#define TYPE1 long long unsigned int
#define TYPE2 long unsigned int
#else
#define TYPE1 size_t
#define TYPE2 size_t 
#endif

// Update this when adding/remove json things
#define CURRENT_VERSION 2
// Update this when modifying classifier parameters
#define CLASSIFIER_VERSION 4

#include <filesystem>
#include <mlpack/mlpack.hpp>
#include "ModelSerializer.h"
#include "Dataset.h"
#include "ClassifierHelper.h"

using namespace mlpack;
using namespace arma;

void PhonemeClassifier::initalize(const size_t& sr) {
    // Check if already initalized
    assert(!initalized);

    initalized = true;

    SAMPLE_RATE = sr;

    ClassifierHelper::instance().initalize(sr);

    outputSize = ClassifierHelper::instance().inversePhonemeSet.size();

    // Load JSON
    bool openedJson = json.open("classifier.json", CURRENT_VERSION);
    if (!openedJson) {
        json["classifier_version"] = CLASSIFIER_VERSION;
        json["input_features"] = (int)inputSize;
        json["output_features"] = (int)outputSize;
        json.save();
    }
    int classifierVersion = json["classifier_version"].get_int();

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
    std::vector<size_t> inputDimensions = { FRAME_SIZE, FFT_FRAMES, 1 };
    int savedInputSize = json["input_features"].get_int();
    int savedOutputSize = json["output_features"].get_int();
    bool metaMatch = (
        classifierVersion == CLASSIFIER_VERSION &&
        savedInputSize == inputSize &&
        savedOutputSize == outputSize);
    if (metaMatch && ModelSerializer::load(&network)) {
        std::cout << "Loaded model\n";
        loaded = true;
    }
    if (!loaded) {
        std::cout << "Model not loaded\n";
        json["classifier_version"] = CLASSIFIER_VERSION;
        json["input_features"] = (int)inputSize;
        json["output_features"] = (int)outputSize;

        network.Add<LinearNoBias>(512);
        network.Add<LeakyReLU>();
        network.Add<LinearNoBias>(512);
        network.Add<LeakyReLU>();
        network.Add<LinearNoBias>(512);
        network.Add<LeakyReLU>();
        network.Add<LinearNoBias>(512);
        network.Add<LeakyReLU>();
        network.Add<Linear>(outputSize);
        network.Add<LogSoftMax>();
    }
    network.InputDimensions() = inputDimensions;
    optimizer.ResetPolicy() = false;

    std::printf("Model initalized with %zd input features and %zd output features\n", inputSize, outputSize);

    ready = loaded;
    std::cout << "Classifier ready\n\n";
}

void PhonemeClassifier::train(const std::string& path, const size_t& examples, const size_t& epochs, const double& stepSize) {
    optimizer.BatchSize() = 128;
    optimizer.StepSize() = stepSize;
    optimizer.MaxIterations() = epochs * examples * outputSize;

    std::vector<Phone> phones;
    
    size_t* phonemeTracker = new size_t[outputSize];
    for (size_t i = 0; i < outputSize; i++) {
        phonemeTracker[i] = 1;
    }

    std::thread trainThread;
    bool isTraining = false;
    Dataset train(path + "/train.tsv", SAMPLE_RATE, path);
    Dataset validate(path + "/dev.tsv", SAMPLE_RATE, path);
    Dataset test(path + "/test.tsv", SAMPLE_RATE, path);
    size_t loops = 0;
    while (true) {
        std::printf("Starting loop %zd\n", loops++);

        train.start(inputSize, outputSize, examples, true);
        validate.start(inputSize, outputSize, examples / 5);
        test.start(inputSize, outputSize, examples / 5);

        if (trainThread.joinable()) {
            trainThread.join();
        }

        train.join();
        validate.join();
        test.join();

        // Start training thread
        bool copyDone = false;
        trainThread = std::thread([this, examples, &train, &validate, &test, &copyDone, loops]{
            mat trainData, trainLabel;
            train.get(trainData, trainLabel);
            mat validateData, validateLabel;
            validate.get(validateData, validateLabel);
            mat testData, testLabel;
            test.get(testData, testLabel);
            copyDone = true;

            // Calculate accuracy
#pragma region Calculate accuracy
            size_t testCount = testLabel.n_cols;
            size_t correctCount = 0;
            size_t* correctPhonemes = new size_t[outputSize];
            size_t** confusionMatrix = new size_t*[outputSize];
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
            for (size_t i = 0; i < testCount; i++) {
                // 80% to skip / 20% to check
                double rnd = (double)rand() / RAND_MAX;
                if (rnd > 0.2) {
                    continue;
                }

                size_t result = classify(testData.submat(span(0, inputSize - 1), span(i, i)));
                size_t label = testLabel(0, i);
                if (result == label) {
                    correctPhonemes[label]++;
                    correctCount++;
                }
                confusionMatrix[label][result]++;
                totalPhonemes[label]++;
                testedExamples++;
            }
            std::printf("Accuracy: %d out of %d (%.1f%%)\n", (int)correctCount, (int)testedExamples, ((double)correctCount / testedExamples) * 100);
            std::cout << "Confusion Matrix:\n";
            std::cout << "   ";
            for (size_t i = 0; i < outputSize; i++) {
                std::cout << std::setw(2) << ClassifierHelper::instance().inversePhonemeSet[i] << " ";
            }
            std::cout << std::endl;
            for (size_t i = 0; i < outputSize; i++) {
                std::cout << std::setw(2) << ClassifierHelper::instance().inversePhonemeSet[i] << " ";
                size_t total = 1;
                for (size_t j = 0; j < outputSize; j++) {
                    total += confusionMatrix[i][j];
                }
                for (size_t j = 0; j < outputSize; j++) {
                    double fraction = (double)confusionMatrix[i][j] / total;
                    int percent = fraction * 100;

                    const char* format = (i == j) ? "\033[36m%2d\033[0m " /* cyan */ :
                        (percent > 0) ? "\033[31m%2d\033[0m " /* red */ : "%2d ";

                    std::printf(format, percent);
                }
                std::cout << "\n";
            }

            delete[] totalPhonemes;
            delete[] correctPhonemes;
            for (size_t i = 0; i < outputSize; i++) {
                delete[] confusionMatrix[i];
            }
            delete[] confusionMatrix;
#pragma endregion

            network.Train(std::move(trainData),
                std::move(trainLabel),
                optimizer,
                ens::PrintLoss(),
                ens::ProgressBar(),
                ens::EarlyStopAtMinLoss());

            json.save();

            ModelSerializer::save(&network, loops);
            });
        
        // Wait to finish copying data into new mat
        while (!copyDone) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
    }
}

size_t PhonemeClassifier::classify(const arma::mat& data) {
    mat results = mat();
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
