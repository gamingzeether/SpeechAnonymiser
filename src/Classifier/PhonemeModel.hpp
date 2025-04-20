#pragma once

#include "../common_inc.hpp"

#include <vector>
#include <string>
#include <zip.h>
#include "ModelSerializer.hpp"
#include "../Utils/ClassifierHelper.hpp"
#include "../Utils/Config.hpp"
#include "../Utils/Global.hpp"
#include "../include_mlpack.hpp"

class PhonemeModel {
public:
	struct Hyperparameters {
		inline static const std::vector<std::string> labels = {
			"Dropout",
			"L2",
			"Batch Size",
			"Step Size",
			"BPTT Steps",
		};
		double& dropout() { return e[0]; };
		double& l2() { return e[1]; };
		double& batchSize() { return e[2]; };
		double& stepSize() { return e[3]; };
		double& bpttSteps() { return e[4]; };

		static const int size = 5;
		double e[size];
	};

	// getInputSize and getSampleRate allow modifiying
	int& getInputSize() { return inputSize; };
	int& getSampleRate() { return sampleRate; };
	int getOutputSize() const { return G_PS_C.size(); };

	NETWORK_TYPE& network() { return net; };
	OPTIMIZER_TYPE& optimizer() { return optim; };
	void setHyperparameters(Hyperparameters hp);
	void initModel();
	void initOptimizer();
	mlpack::NegativeLogLikelihoodWType<MAT_TYPE>& outputLayer() { return net.OutputLayer(); };

	void save(int checkpoint = -1);
	bool load();
private:
	void cleanUnpacked();
	void logZipError(int error);
	void logZipError(zip_t* archive);
	void logZipError(zip_error_t* error);
	void setDefaultModel();
	void addConv(JSONHelper::JSONObj& layers, int maps, int width, int height, int strideX, int strideY);
	void addPooling(JSONHelper::JSONObj& layers, int width, int height, int strideX, int strideY);
	void addLinear(JSONHelper::JSONObj& layers, int neurons);
	void addLstm(JSONHelper::JSONObj& layers, int neurons);
	std::string getTempPath();

	NETWORK_TYPE net;
	OPTIMIZER_TYPE optim;
	Config config;
	Hyperparameters hp;

	int inputSize = FRAME_SIZE * FFT_FRAMES * 3;
	int outputSize = 0;
	int sampleRate = 0;
};
