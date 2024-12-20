#pragma once

#include "common_inc.h"

#include <string>

class Util {
public:
    // https://stackoverflow.com/a/7154226
    static std::wstring utf8_to_utf16(const std::string& utf8);
    static size_t customHasher(const std::wstring& str);
    static void removeTrailingSlash(std::string& path);
};
