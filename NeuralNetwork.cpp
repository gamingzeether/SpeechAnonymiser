#include "NeuralNetwork.h"

void NeuralNetwork::process(const float* in, const size_t& inSize, OUT float* out, const size_t& outSize) {
	for (size_t i = 0; i < inSize; i++) {
		layers[0].set(i, 0, in[i]);
	}
	for (int i = 0; i < neuronLayers.size(); i++) {
		neuronLayers[i].matrix_multiply(layers[i], layers[i + 1]);
	}
	Matrix& outputLayer = layers[layers.size() - 1];
	for (size_t i = 0; i < outSize; i++) {
		out[i] = outputLayer.get(i, 0);
	}
}

NeuralNetwork::NeuralNetwork(std::vector<size_t>& neuronCount) {
	layers = std::vector<Matrix>(neuronCount.size());
	neuronLayers = std::vector<Matrix>(neuronCount.size() - 1);
	for (int i = 0; i < neuronCount.size(); i++) {
		layers[i] = Matrix(neuronCount[i], 1);
	}
	for (int i = 0; i < neuronCount.size() - 1; i++) {
		size_t rows = neuronCount[i + 1];
		size_t columns = neuronCount[i];
		neuronLayers[i] = Matrix(rows, columns);
		// Fill matrix with 1s
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < columns; c++) {
				neuronLayers[i].set(r, c, 1.0f);
			}
		}
	}
}
