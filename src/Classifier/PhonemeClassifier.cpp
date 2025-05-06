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
#define CONVERT(mat) CUBE_TYPE CNAME(mat) = coot::conv_to<CUBE_TYPE>::from(mat)
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
  hp.stepSize() = 1e-3;
  hp.bpttSteps() = 50;
  model.setHyperparameters(hp);

  model.getSampleRate() = sampleRate;

  if (!model.load()) {
    G_LG("Model not loaded, initalizing new model", Logger::WARN, Logger::YELLOW);
  }

  G_LG(Util::format("Model initalized with %d input features and %d output features", CLASSIFIER_ROW_SIZE, model.getOutputSize()), Logger::INFO);

  ready = true;
  G_LG("Classifier ready", Logger::INFO, Logger::GREEN);
}

void PhonemeClassifier::train(const std::string& path, const size_t& examples, const size_t& epochs) {
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
  
  int inputSize = CLASSIFIER_ROW_SIZE;
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
  arma::urowvec testLengths, trainLengths, validateLengths;
  CPU_CUBE_TYPE testData, testLabel, trainData, trainLabel, validateData, validateLabel;
  test.get(testData, testLabel, testLengths);
  train.get(trainData, trainLabel, trainLengths);
  validate.get(validateData, validateLabel, validateLengths);
  int epoch = 0;

  CONVERT(trainData);
  CONVERT(trainLabel);
  CONVERT(validateData);
  CONVERT(validateLabel);

  model.optimizer().MaxIterations() = epochs * trainLabel.n_cols;

  //AutoregressivePredictiveCoding apc;
  //apc.train(trainData); 
  
  // Set class weights
  model.outputLayer().ClassWeights() = weighClasses(trainLabel, trainLengths);

  // Start training model
  int numSlices = std::accumulate(trainLengths.begin(), trainLengths.end(), 0);
  G_LG(Util::format("Total number of time slices: %d", numSlices), Logger::INFO);
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

// Note: This doesn't reset memory, only use for inference and not evaluating
size_t PhonemeClassifier::classify(const CUBE_TYPE& data) {
  CUBE_TYPE results;
  model.network().PredictSingle(data, results);
  ELEM_TYPE max = results(0);
  size_t maxIdx = 0;
  for (size_t i = 0; i < model.getOutputSize(); i++) {
    ELEM_TYPE val = results(i);
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
  int inputSize = CLASSIFIER_ROW_SIZE;
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
    std::cout << Util::leftPad(phoneme, 3) << " ";
  }
  std::cout << std::endl;
  for (size_t label = 0; label < outputSize; label++) {
    std::string phoneme = ps.xSampa(label);
    std::cout << Util::leftPad(phoneme, 3) << " ";
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

void PhonemeClassifier::tuneHyperparam(const std::string& path, int iterations, int mr) {
  int inputSize = CLASSIFIER_ROW_SIZE;
  int outputSize = model.getOutputSize();

  // Get data
  arma::urowvec trainLengths, validLengths;
  CPU_CUBE_TYPE trainData, trainLabel, validData, validLabel;
  {
    Dataset train(16000, path);
    Dataset validate(16000, path);
    train.setSubtype(Subtype::TRAIN);
    validate.setSubtype(Subtype::VALIDATE);
    size_t tuneSize = 250;
    train.start(inputSize, outputSize, tuneSize * 0.8, true);
    validate.start(inputSize, outputSize, tuneSize * 0.2, true);
    train.join();
    validate.join();
    train.get(trainData, trainLabel, trainLengths);
    validate.get(validData, validLabel, validLengths);
  }

  CONVERT(trainData);
  CONVERT(trainLabel);
  CONVERT(validData);
  CONVERT(validLabel);

  // Prepare models
  // mr is the number of steps in each direction
  int numModels = mr * 2 + 1;
  std::vector<PhonemeModel> models(numModels);
  std::vector<float> accuracies(numModels);
  std::vector<std::thread> trainThreads(numModels);

  // Initalize parameters
  PhonemeModel::Hyperparameters base = PhonemeModel::Hyperparameters();
  base.dropout() = 0.3;
  base.l2() = 0.1;
  base.batchSize() = 1;
  base.stepSize() = 0.005;
  base.bpttSteps() = 40;

  // Per "step" distance; 1 steps, delta = 1*stepSize; 2 steps, delta = 2*stepSize etc
  PhonemeModel::Hyperparameters stepSize = PhonemeModel::Hyperparameters();
  stepSize.dropout() = 0.05;
  stepSize.l2() = 0.05;
  stepSize.batchSize() = -1;
  stepSize.stepSize() = 0.0001;
  stepSize.bpttSteps() = 5;

  PhonemeModel::Hyperparameters min = PhonemeModel::Hyperparameters();
  min.dropout() = 0.05;
  min.l2() = 0.05;
  min.batchSize() = -1;
  min.stepSize() = 0.0001;
  min.bpttSteps() = 20;

  PhonemeModel::Hyperparameters max = PhonemeModel::Hyperparameters();
  max.dropout() = 0.5;
  max.l2() = 1.00;
  max.batchSize() = -1;
  max.stepSize() = 0.01;
  max.bpttSteps() = 200;

  int paramSize = PhonemeModel::Hyperparameters::size;

  int optimEpochs = 10;
  int optimIterations = optimEpochs * trainLabel.n_cols;

  auto classWeights = weighClasses(trainLabel, trainLengths);

  // Do iterations
  // Each iteration adjusts one parameter up or down various amounts per model
  // Then continues with the best out of them
  for (int i = 0; i < iterations * paramSize; i++) {
    int iteration = i / paramSize;
    int targetParam = i % paramSize;
    float delta = stepSize.e[targetParam];

    if (delta < 0) {
      G_LG(Util::format("Skipping subiteration %d", targetParam), Logger::INFO);
      continue;
    }
    G_LG(Util::format("Starting iteration %d, subiteration %d", iteration, targetParam), Logger::INFO);
    for (int j = 0; j < models.size(); j++) {
      // Initalize models with parameters
      PhonemeModel::Hyperparameters tempParams = base;
      // Check to make sure change is within bounds
      tempParams.e[targetParam] += delta * (j - mr);
      if (tempParams.e[targetParam] <= min.e[targetParam] || tempParams.e[targetParam] >= max.e[targetParam]) {
        G_LG(Util::format("Skipping %d, out of bounds (%lf)", j, tempParams.e[targetParam]), Logger::DBUG);
        continue;
      }

      {
        PhonemeModel& mdl = models[j];
        // Reset model
        Global::supressLog(true); // Hide the initalization messages
        mdl = PhonemeModel();
        mdl.setHyperparameters(tempParams);
        mdl.getSampleRate() = 16000;
        mdl.load();
        Global::supressLog(false);
        mdl.network().OutputLayer().ClassWeights() = classWeights;
        // Make sure every model starts at the same place
        if (j > 1) {
          auto copy = models[0].network().Parameters();
          mdl.network().Parameters() = copy;
        }
      }

      // Start training models
      G_LG(Util::format("Starting model %d", j), Logger::INFO);
      std::thread& trainThread = trainThreads[j];
      trainThread = std::thread([this,
                     &models,
                     &CNAME(trainData),
                     &CNAME(trainLabel),
                     &trainLengths,
                     &CNAME(validData),
                     &CNAME(validLabel),
                     &validLengths,
                     &accuracies,
                     j,
                     optimIterations]
      {
        PhonemeModel& mdl = models[j];
        mdl.optimizer().MaxIterations() = optimIterations;

        mdl.network().Train(
          CNAME(trainData),
          CNAME(trainLabel),
          trainLengths,
          mdl.optimizer(),
          TrainingExecType<MAT_TYPE>(
            [&](size_t epoch) {
              mdl.network().SetNetworkMode(false);
              double acc = accuracy(mdl.network(), CNAME(validData), CNAME(validLabel), validLengths);
              mdl.network().SetNetworkMode(true);
              accuracies[j] = acc;
              G_LG(Util::format("Model %d epoch %ld: %lf", j, epoch, acc), Logger::INFO);
            }
          )
        );
        G_LG(Util::format("Model %d done with acc: %lf", j, accuracies[j]), Logger::INFO);
      });
    }

    // Wait for training to complete
    for (int j = 0; j < numModels; j++) {
      if (trainThreads[j].joinable()) {
        trainThreads[j].join();
      }
    }

    // Evaluate best
    int bestIndex = -1;
    float bestAcc = -1;
    for (int j = 0; j < numModels; j++) {
      float acc = accuracies[j];
      if (acc > bestAcc) {
        bestAcc = acc;
        bestIndex = j;
      }
    }
    float bestDelta = delta * (bestIndex - mr);
    double targetBefore = base.e[targetParam];
    base.e[targetParam] += bestDelta;
    double targetAfter = base.e[targetParam];

    // Print new best
    G_LG(Util::format("Best hyperparameters after iteration %d, subiteration %d", iteration, targetParam), Logger::INFO);
    for (int j = 0; j < paramSize; j++) {
      if (j == targetParam) {
        G_LG(Util::format("  %s: %lf <- %lf", base.labels[j].c_str(), targetAfter, targetBefore), Logger::INFO);
      } else {
        G_LG(Util::format("  %s: %lf", base.labels[j].c_str(), base.e[j]), Logger::INFO);
      }
    }
  }
}

void PhonemeClassifier::evaluate(const std::string& path) {
  int inputSize = CLASSIFIER_ROW_SIZE;
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

double PhonemeClassifier::accuracy(NETWORK_TYPE& network, const CPU_CUBE_TYPE& data, const CPU_CUBE_TYPE& labels, const arma::urowvec& lengths) {
  CPU_CUBE_TYPE predictions;
  network.Predict(data, predictions, lengths);
  size_t correctCount = 0, totalCount = 0;
  for (size_t col = 0; col < labels.n_cols; col++) {
    size_t seqLen = lengths[col];
    for (size_t s = 0; s < seqLen; s++) {
      size_t label = labels(0, col, s);
      size_t prediction = predictions.slice(s).col(col).index_max();
      if (label == prediction)
        correctCount++;
      totalCount++;
    }
  }
  return (double)correctCount / totalCount;
}

arma::Row<MAT_TYPE::elem_type> PhonemeClassifier::weighClasses(const CPU_CUBE_TYPE& labels, const arma::urowvec& lengths) {
  size_t outputSize = model.getOutputSize();
  std::vector<size_t> labelCounts(outputSize);
  for (size_t col = 0; col < labels.n_cols; col++) {
    size_t length = lengths[col];
    for (size_t slice = 0; slice < length; slice++) {
      labelCounts[labels(0, col, slice)]++;
    }
  }

  size_t nPoints = std::accumulate(labelCounts.begin(), labelCounts.end(), 0);
  arma::Row<MAT_TYPE::elem_type> outWeights(outputSize);
  for (size_t i = 0; i < outputSize; i++) {
    outWeights(i) = (double)nPoints / (outputSize * labelCounts[i]);
  }
  return outWeights;
}
