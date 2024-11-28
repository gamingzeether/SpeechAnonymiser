#pragma once

#include "common_inc.h"

#include <string>

class ModelSerializer
{
public:
	static void saveNetwork(const std::string& filename, const void* network);
	static bool loadNetwork(const std::string& filename, void* network);
};
