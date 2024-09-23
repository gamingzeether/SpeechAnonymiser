#include "ModelSerializer.h"

#define MLPACK_ENABLE_ANN_SERIALIZATION 
#include <filesystem>
#include <mlpack/mlpack.hpp>
CEREAL_REGISTER_TYPE(mlpack::LinearType<arma::mat, mlpack::L2Regularizer>);

#define NETWORK_TYPE mlpack::FFN<mlpack::NegativeLogLikelihood, mlpack::RandomInitialization>
using namespace mlpack;

void ModelSerializer::save(const void* network, int checkpoint) {
	const NETWORK_TYPE& netRef = *(NETWORK_TYPE*)network;
	data::Save(MODEL_FILE + MODEL_EXT, "model", netRef);
	if (checkpoint >= 0) {
		std::filesystem::copy_file(
			MODEL_FILE + MODEL_EXT,
			MODEL_FILE + MODEL_EXT + std::to_string(checkpoint),
			std::filesystem::copy_options::overwrite_existing);
	}
}

bool ModelSerializer::load(void* network) {
	NETWORK_TYPE& netRef = *(NETWORK_TYPE*)network;
	return data::Load(MODEL_FILE + MODEL_EXT, "model", netRef);
}
