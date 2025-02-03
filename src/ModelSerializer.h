#pragma once

#include "common_inc.h"

#include <string>

#define NETWORK_TYPE mlpack::FFN<mlpack::NegativeLogLikelihoodType<MAT_TYPE>, mlpack::HeInitialization, MAT_TYPE>
#define OPTIMIZER_TYPE ens::AdaBelief

class ModelSerializer
{
public:
	static void saveNetwork(const std::string& filename, const void* network);
	static bool loadNetwork(const std::string& filename, void* network);
};
