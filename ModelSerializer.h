#pragma once

#include "common_inc.h"

#include <string>

class ModelSerializer
{
public:
	static void save(const void* network, int checkpoint = -1);
	static bool load(void* network);
private:
	inline static const std::string MODEL_FILE = "phoneme_model";
	inline static const std::string MODEL_EXT = ".bin";
};
