#pragma once

#include "common_inc.h"

#include "include_mlpack.h"
#include "ClassifierHelper.h"
#include "Config.h"

#define NETWORK_TYPE mlpack::FFN<mlpack::NegativeLogLikelihoodType<MAT_TYPE>, mlpack::RandomInitialization, MAT_TYPE>
#define OPTIMIZER_TYPE ens::Adam

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

	void save(int checkpoint = -1);
	bool load();
private:
	void cleanUnpacked();

	NETWORK_TYPE net;
	OPTIMIZER_TYPE optim;
	Config config;
	Hyperparameters hp;

	int inputSize = FRAME_SIZE * FFT_FRAMES;
	int outputSize = 0;
	int sampleRate = 0;
};
