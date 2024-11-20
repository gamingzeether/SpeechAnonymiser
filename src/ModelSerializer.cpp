#include "ModelSerializer.h"

#define MLPACK_ENABLE_ANN_SERIALIZATION 
#include <filesystem>
#include "include_mlpack.h"
CEREAL_REGISTER_MLPACK_LAYERS(MAT_TYPE);
CEREAL_REGISTER_TYPE(mlpack::LinearType<MAT_TYPE, mlpack::L2Regularizer>);
CEREAL_REGISTER_TYPE(mlpack::LinearNoBiasType<MAT_TYPE, mlpack::L2Regularizer>);

#define NETWORK_TYPE mlpack::FFN<mlpack::NegativeLogLikelihoodType<MAT_TYPE>, mlpack::RandomInitialization, MAT_TYPE>

void ModelSerializer::save(const void* network, int checkpoint) {
	const NETWORK_TYPE& netRef = *(NETWORK_TYPE*)network;
	mlpack::data::Save(MODEL_FILE + MODEL_EXT, "model", netRef, true);
	if (checkpoint >= 0) {
		std::filesystem::copy_file(
			MODEL_FILE + MODEL_EXT,
			MODEL_FILE + MODEL_EXT + std::to_string(checkpoint),
			std::filesystem::copy_options::overwrite_existing);
	}
}

bool ModelSerializer::load(void* network) {
	NETWORK_TYPE& netRef = *(NETWORK_TYPE*)network;
	return mlpack::data::Load(MODEL_FILE + MODEL_EXT, "model", netRef);
}
