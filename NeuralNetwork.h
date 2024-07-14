#pragma once
#include "define.h"

#define OUT

#include <vector>
#include "Matrix.h"

class NeuralNetwork {
public:
	void process(const float* in, const size_t& inSize, OUT float* out, const size_t& outSize);

	NeuralNetwork(std::vector<size_t>& neuronCount);
private:
	std::vector<Matrix> neuronLayers;
	std::vector<Matrix> layers;
};
