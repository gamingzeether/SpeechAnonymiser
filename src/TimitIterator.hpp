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
private:
    size_t pointer = 0;
    std::vector<std::filesystem::path> paths;
};
