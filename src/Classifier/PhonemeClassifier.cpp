#include "PhonemeClassifier.hpp"

#ifdef __GNUC__
#define TYPE1 long long unsigned int
#define TYPE2 long unsigned int
#else
#define TYPE1 size_t
#define TYPE2 size_t 
#endif

#ifdef USE_GPU
#define CNAME(mat) gpu##mat
#define CONVERT(mat) MAT_TYPE CNAME(mat) = coot::conv_to<MAT_TYPE>::from(mat)
#else
#define CNAME(mat) mat
#define CONVERT(mat) ;
#endif

#include <filesystem>
#include "ModelSerializer.hpp"
#include "TrainingExec.hpp"
#include "Train/Dataset.hpp"
#include "Train/eta_callback.hpp"
#include "../Utils/ClassifierHelper.hpp"
#include "../Utils/Global.hpp"
#include "../Utils/Util.hpp"

using namespace mlpack;

void PhonemeClassifier::initalize(const size_t& sr) {
    // Check if already initalized
    assert(!initalized);

    initalized = true;

    sampleRate = sr;

    PhonemeModel::Hyperparameters hp = PhonemeModel::Hyperparameters();
    hp.dropout() = 0.25;
    hp.l2() = 0.01;
    // Batch size must be 1 for ragged sequences
    hp.batchSize() = 1;
    hp.stepSize() = 0.0001;
    hp.bpttSteps() = 40;
    model.setHyperparameters(hp);

    model.getSampleRate() = sampleRate;

    if (!model.load()) {
        G_LG("Model not loaded, initalizing new model", Logger::WARN, Logger::YELLOW);
    }

    G_LG(Util::format("Model initalized with %d input features and %d output features", model.getInputSize(), model.getOutputSize()), Logger::INFO);

    ready = true;
    G_LG("Classifier ready", Logger::INFO, Logger::GREEN);
}

void PhonemeClassifier::train(const std::string& path, const size_t& examples, const size_t& epochs) {
    //tuneHyperparam(path, 100);

    // Set number of examples to be a multiple of batch size
    size_t realExamples;
    {
        size_t bs = model.optimizer().BatchSize();
        size_t nSteps = std::max((size_t)1, examples / bs);
        realExamples = nSteps * bs;
        if (realExamples != examples) {
            G_LG(Util::format("Rounding examples to %ld (multiple of %ld)", realExamples, bs), Logger::INFO);
        }
    }
    
    int inputSize = model.getInputSize();
    int outputSize = model.getOutputSize();

    const bool trainGeneral = true;
    std::string clientFilter = (trainGeneral) ? "" :
        // Train model on specific speaker
        // This client has a lot of entries and is has american english which is what the aligner used
        "b419faab633f2099c6405ff157b4d9fb5675219570f2683a4d08cbadeac4431e9d9b30dfa9b04f79aad9d8e3f75fda964809f3aa72ae9d0a4a025c59417f3dd1";
    Dataset train(sampleRate, path, clientFilter);
    Dataset validate(sampleRate, path);
    Dataset test(sampleRate, path);
    train.setSubtype(Subtype::TRAIN);
    validate.setSubtype(Subtype::VALIDATE);
    test.setSubtype(Subtype::TEST);
    double bestLoss = 9e+99;

    // Start loading data
    train.start(inputSize, outputSize, realExamples, model.optimizer().BatchSize(), true);
    test.start(inputSize, outputSize, realExamples / 10);
    validate.start(inputSize, outputSize, realExamples / 10);

    // Wait to finish loading
    while (!train.done() || !test.done() || !validate.done()) {
        std::string status = Util::format("Train: %ld, Test: %ld, Validate: %ld\r",
                train.getLoadedClips(),
                test.getLoadedClips(),
                validate.getLoadedClips());
        std::cout << status << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    train.join();
    test.join();
    validate.join();

    // Prepare data for training
    arma::urowvec testLengths, trainLengths, validationLengths;
    CPU_CUBE_TYPE testData, testLabel, trainData, trainLabel, validateData, validateLabel;
    test.get(testData, testLabel, testLengths);
    train.get(trainData, trainLabel, trainLengths);
    validate.get(validateData, validateLabel, validationLengths);
    int epoch = 0;

    CONVERT(trainData);
    CONVERT(trainLabel);
    CONVERT(validateData);
    CONVERT(validateLabel);

    model.optimizer().MaxIterations() = epochs * trainLabel.n_cols;

    /* Dataset debugging code
    {
        std::vector<std::string> imageNames;
        CPU_MAT_TYPE images = trainData;
        for (size_t i = 0; i < images.n_cols; i++) {
            size_t phone = trainLabel[i];
            std::string folder = Util::format("debug/data/%ld/", phone);
            if (!std::filesystem::exists(folder))
                std::filesystem::create_directories(folder);
            imageNames.push_back(Util::format("%s/%ld.png", folder.CS, i));

            auto col = images.col(i);
            float min = col.min();
            //float max = col.max();
            //float range = max - min;
            //col -= min;
            //col *= 255.0f / range;
        }
        // The saved image is actually flipped both horizontally and vertically
        data::ImageInfo imageInfo = data::ImageInfo(FFT_FRAMES, FRAME_SIZE, 3);
        data::Save(imageNames, images, imageInfo);
    }

    {
        std::cout << trainData.max() << "\n";
        std::cout << trainData.min() << "\n";
        std::string str;
        std::getline(std::cin, str);
        for (size_t i = 0; i < trainData.n_cols; i++) {
            std::cout << trainData.col(i) << "\n";
            std::cout << trainLabel.col(i) << "\n";
            std::getline(std::cin, str);
        }
    }
    */
    
    // Set class weights
    {
        std::vector<size_t> labelCounts(outputSize);
        for (size_t col = 0; col < trainLabel.n_cols; col++) {
            for (size_t slice = 0; slice < trainLabel.n_slices; slice++) {
                labelCounts[trainLabel(0, col, slice)]++;
            }
        }
        size_t nPoints = std::accumulate(labelCounts.begin(), labelCounts.end(), 0);
        arma::Row<MAT_TYPE::elem_type>& classWeights = model.outputLayer().ClassWeights();
        if (classWeights.n_elem < outputSize) {
            classWeights.ones(outputSize);
        }
        for (size_t i = 0; i < outputSize; i++) {
            classWeights(i) = (double)nPoints / (outputSize * labelCounts[i]);
        }
    }

    // Start training model
    int numPoints = std::accumulate(trainLengths.begin(), trainLengths.end(), 0);
    G_LG(Util::format("Total number of training points: %d", numPoints), Logger::INFO);
    G_LG("Starting training", Logger::INFO);
    model.network().Train(
        CNAME(trainData),
        CNAME(trainLabel),
        trainLengths,
        model.optimizer(),
        ens::ProgressBarETA(50),
        TrainingExecType<MAT_TYPE>(
            [&](size_t epoch) {
                //G_LG(Util::format("Finished epoch %ld with learning rate %lf", epoch, model.optimizer().StepSize()), Logger::DBUG);
                model.network().SetNetworkMode(false);
                // Compare training and test accuracy to check for overfitting
                if (epoch % 10 == 0) {
                    printConfusionMatrix(trainData, trainLabel, trainLengths);
                    printConfusionMatrix(testData, testLabel, testLengths);
                }

                model.save(epoch);
                model.network().SetNetworkMode(true);
            }
        ));

        G_LG(Util::format("Training ended with best loss %lf", bestLoss), Logger::INFO);
}

// Note: This doesn't reset memory
size_t PhonemeClassifier::classify(const CUBE_TYPE& data) {
    CUBE_TYPE results;
    model.network().PredictSingle(data, results);
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
    return G_PS_C.xSampa(in);
};

void PhonemeClassifier::printConfusionMatrix(const CPU_CUBE_TYPE& testData, const CPU_CUBE_TYPE& testLabel, const arma::urowvec& lengths) {
    int inputSize = model.getInputSize();
    int outputSize = model.getOutputSize();

    size_t correctCount = 0;
    std::vector<size_t> correctPhonemes(outputSize);
    std::vector<std::vector<size_t>> confusionMatrix(outputSize);
    std::vector<size_t> totalPhonemes(outputSize);
    for (size_t i = 0; i < outputSize; i++) {
        correctPhonemes[i] = 0;
        totalPhonemes[i] = 0;
        confusionMatrix[i] = std::vector<size_t>(outputSize, 0);
    }
    size_t testedExamples = 0;

    CONVERT(testData);

    CUBE_TYPE results;
    model.network().Predict(CNAME(testData), results);
    {
        for (size_t c = 0; c < results.n_cols; c++) {
            size_t length = lengths(c);
            for (size_t s = 0; s < length; s++) {
                size_t maxIdx = results.slice(s).col(c).index_max();
                size_t label = testLabel(0, c, s);
                if (maxIdx == label) {
                    correctPhonemes[label]++;
                    correctCount++;
                }
                confusionMatrix[label][maxIdx]++;
                totalPhonemes[label]++;
                testedExamples++;
            }
        }
    }
    G_LG(Util::format("Accuracy: %d out of %d (%.2lf%%)", (int)correctCount, (int)testedExamples, ((double)correctCount / testedExamples) * 100), Logger::INFO);
    std::cout << "Confusion Matrix:\n";
    std::cout << "    ";
    const PhonemeSet& ps = G_PS_C;
    for (size_t i = 0; i < outputSize; i++) {
        std::string phoneme = ps.xSampa(i);
        Util::leftPad(phoneme, 3);
        std::cout << phoneme << " ";
    }
    std::cout << std::endl;
    for (size_t label = 0; label < outputSize; label++) {
        std::string phoneme = ps.xSampa(label);
        Util::leftPad(phoneme, 3);
        std::cout << phoneme << " ";
        size_t total = totalPhonemes[label];
        for (size_t prediction = 0; prediction < outputSize; prediction++) {
            std::string format;
            double fraction = (double)confusionMatrix[label][prediction] / total;
            int percent = fraction * 100;
            if (total == 0)
                percent = 0;
            
            if (label == prediction) {
                if (percent == 100) {
                    format = "\033[32m%3d\033[0m "; /* 100% accuracy: green */
                } else {
                    format = "\033[36m%3d\033[0m "; /* diagonal: cyan */
                }
            } else {
                if (percent > 0) {
                    format = "\033[31m%3d\033[0m "; /* >= 1% misclassify: red */
                } else {
                    format = "%3d " /* 0% misclassify: white */;
                }
            }
            
            std::printf(format.c_str(), percent % 100);
        }

        std::cout << "\n";
    }
}

void PhonemeClassifier::tuneHyperparam(const std::string& path, int iterations) {
    int inputSize = model.getInputSize();
    int outputSize = model.getOutputSize();

    // Get data
    Dataset train(16000, path);
    Dataset validate(16000, path);
    train.setSubtype(Subtype::TRAIN);
    validate.setSubtype(Subtype::VALIDATE);
    size_t tuneSize = 250;
    train.start(inputSize, outputSize, tuneSize * 0.8, true);
    validate.start(inputSize, outputSize, tuneSize * 0.2, true);
    train.join();
    validate.join();
    arma::urowvec trainLengths, validLengths;
    CPU_CUBE_TYPE tuneTrainData, tuneTrainLabel, tuneValidData, tuneValidLabel;
    train.get(tuneTrainData, tuneTrainLabel, trainLengths);
    validate.get(tuneValidData, tuneValidLabel, validLengths);

    CONVERT(tuneTrainData);
    CONVERT(tuneTrainLabel);
    CONVERT(tuneValidData);
    CONVERT(tuneValidLabel);

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
    base.batchSize() = 1;
    base.stepSize() = 0.005;
    base.bpttSteps() = 40;

    PhonemeModel::Hyperparameters stepSize = PhonemeModel::Hyperparameters();
    stepSize.dropout() = 0.05;
    stepSize.l2() = 0.0001;
    stepSize.batchSize() = 0;
    stepSize.stepSize() = 0.0001;
    stepSize.bpttSteps() = 5;

    int paramSize = PhonemeModel::Hyperparameters::size;

    int optimEpochs = 500;
    int optimIterations = optimEpochs * (tuneSize * 0.8) * outputSize;

    // Do iterations
    // Each iteration adjusts one parameter up or down various amounts per model
    // Then continues with the best out of them
    for (int i = 0; i < iterations * paramSize; i++) {
        int targetParam = i % paramSize;
        float delta = stepSize.e[targetParam];

        G_LG(Util::format("Starting iteration %d, subiteration %d", i / paramSize, i % paramSize), Logger::INFO);
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
            G_LG(Util::format("Starting model %d", j), Logger::INFO);
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
                            double validationLoss = 1; //mdl.network().Evaluate(CNAME(tuneValidData), CNAME(tuneValidLabel));
                            if (validationLoss < bestLosses[j]) {
                                bestLosses[j] = validationLoss;
                                G_LG(Util::format("Model %d with new best %lf", j, validationLoss), Logger::INFO);
                            }
                            return validationLoss;
                        })
                );
                G_LG(Util::format("Model %d done", j), Logger::INFO);
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
        G_LG(Util::format("Best hyperparameters after iteration %d, subiteration %d", i / paramSize, i % paramSize), Logger::INFO);
        for (int j = 0; j < paramSize; j++) {
            G_LG(Util::format("%lf", base.e[j]), Logger::INFO);
        }
    }
}

void PhonemeClassifier::evaluate(const std::string& path) {
    int inputSize = model.getInputSize();
    int outputSize = model.getOutputSize();

    Dataset test(16000, path);
    test.setSubtype(Subtype::TEST);
    test.start(inputSize, outputSize, 500, true);
    test.join();

    arma::urowvec lengths;
    CPU_CUBE_TYPE data, labels;
    test.get(data, labels, lengths);
    printConfusionMatrix(data, labels, lengths);
}
