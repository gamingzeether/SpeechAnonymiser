#pragma once

#include "../common_inc.hpp"

#include "../include_mlpack.hpp"

#include "ModelSerializer.hpp"

#define _APC_ENCODER_NET_TYPE mlpack::FFN<mlpack::CosineEmbeddingLossType<MAT_TYPE>, mlpack::RandomInitialization, MAT_TYPE>

// T-APC described in paper https://arxiv.org/abs/1910.12607
class AutoregressivePredictiveCoding {
public:
  void train(const CUBE_TYPE& data);
  void encode(const MAT_TYPE& in, MAT_TYPE& out);
private:
  void addEncoderLayers(_APC_ENCODER_NET_TYPE& network, size_t embDim, size_t sequenceLength);
  void flatten(const CUBE_TYPE& in, MAT_TYPE& out);

  _APC_ENCODER_NET_TYPE encoder;
};
