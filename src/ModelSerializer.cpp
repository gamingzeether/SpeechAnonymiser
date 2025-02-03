#include "ModelSerializer.h"

#define MLPACK_ENABLE_ANN_SERIALIZATION 
#include <filesystem>
#include "include_mlpack.h"
CEREAL_REGISTER_MLPACK_LAYERS(MAT_TYPE);
CEREAL_REGISTER_TYPE(mlpack::LinearType<MAT_TYPE, mlpack::L2Regularizer>);
CEREAL_REGISTER_TYPE(mlpack::LinearNoBiasType<MAT_TYPE, mlpack::L2Regularizer>);

void ModelSerializer::saveNetwork(const std::string& filename, const void* network) {
	const NETWORK_TYPE& netRef = *(NETWORK_TYPE*)network;
	mlpack::data::Save(filename, "model", netRef, true);
}

bool ModelSerializer::loadNetwork(const std::string& filename, void* network) {
	NETWORK_TYPE& netRef = *(NETWORK_TYPE*)network;
	return mlpack::data::Load(filename, "model", netRef);
}
