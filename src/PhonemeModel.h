#pragma once

#include "common_inc.h"

#include "include_mlpack.h"
#include "ClassifierHelper.h"
#include "Config.h"
#include "Logger.h"
#include <zip.h>

#define NETWORK_TYPE mlpack::FFN<mlpack::NegativeLogLikelihoodType<MAT_TYPE>, mlpack::HeInitialization, MAT_TYPE>
#define OPTIMIZER_TYPE ens::AdaBelief

class PhonemeModel {
public:
	struct Hyperparameters {
		float& dropout() { return e[0]; };
		float& l2() { return e[1]; };
		float& batchSize() { return e[2]; };
		float& stepSize() { return e[3]; };
		float& warmup() { return e[4]; };

		static const int size = 5;
		float e[size];
	};

	int& getInputSize() { return inputSize; };
	int& getOutputSize() { return outputSize; };
	int& getSampleRate() { return sampleRate; };

	NETWORK_TYPE& network() { return net; };
	OPTIMIZER_TYPE& optimizer() { return optim; };
	float rate(int epoch) { return (epoch < hp.warmup()) ? (epoch / hp.warmup()) * hp.stepSize() : hp.stepSize(); };
	void setHyperparameters(Hyperparameters hp);
	void initModel();
	void initOptimizer();
	void useLogger(Logger& l) { logger = l; };

	void save(int checkpoint = -1);
	bool load();
private:
	void cleanUnpacked();
	void logZipError(int error);
	void logZipError(zip_t* archive);
	void logZipError(zip_error_t* error);

	NETWORK_TYPE net;
	OPTIMIZER_TYPE optim;
	Config config;
	Hyperparameters hp;
	std::optional<Logger> logger;

	int inputSize = FRAME_SIZE * FFT_FRAMES * 3;
	int outputSize = 0;
	int sampleRate = 0;
};
