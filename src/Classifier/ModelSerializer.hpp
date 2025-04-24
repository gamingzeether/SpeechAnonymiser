#pragma once

#include "../common_inc.hpp"

#include <string>

#define NETWORK_TYPE mlpack::RNN<mlpack::NegativeLogLikelihoodWType<MAT_TYPE>, mlpack::RandomInitialization, MAT_TYPE>
#define OPTIMIZER_TYPE ens::RMSProp

class ModelSerializer
{
public:
  static void saveNetwork(const std::string& filename, const void* network);
  static bool loadNetwork(const std::string& filename, void* network);
};
