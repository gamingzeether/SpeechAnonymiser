#pragma once

#include "common_inc.h"

#include "include_mlpack.h"
#include "ClassifierHelper.h"

#define NETWORK_TYPE mlpack::FFN<mlpack::NegativeLogLikelihoodType<MAT_TYPE>, mlpack::RandomInitialization, MAT_TYPE>
#define OPTIMIZER_TYPE ens::Adam

class PhonemeModel {
public:
	struct Hyperparameters {
		float& dropout() { return e[0]; };
		float& l2() { return e[1]; };
		float& batchSize() { return e[2]; };
		float& stepSize() { return e[3]; };

		static const int size = 4;
		float e[size];
	};

	NETWORK_TYPE& network() { return net; };
	OPTIMIZER_TYPE& optimizer() { return optim; };
	void initModel(Hyperparameters hp);
private:
	NETWORK_TYPE net;
	OPTIMIZER_TYPE optim;
};
