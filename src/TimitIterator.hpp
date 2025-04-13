#pragma once

#include "common_inc.hpp"

#include <vector>
#include <string>
#include <filesystem>

class TimitIterator {
public:
    void open(const std::string& path);
    std::filesystem::path next();
    bool good();
    void resetCounter() { pointer = 0; };
private:
    size_t pointer = 0;
    std::vector<std::filesystem::path> paths;
};
