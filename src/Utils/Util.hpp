#pragma once

#include "../common_inc.hpp"

#include <vector>
#include <string>

class Util {
public:
    // https://stackoverflow.com/a/7154226
    static std::wstring utf8_to_utf16(const std::string& utf8);
    // Hashes a wstring
    static size_t customHasher(const std::wstring& str);
    // Removes a trailing slash from a path
    static void removeTrailingSlash(std::string& path);
    // Returns true if changed (padding was added)
    static bool leftPad(std::string& message, int width, const char padChar = ' ');
    // Split a string into parts seperated by delimiter (delimiter is removed)
    static std::vector<std::string> split(const std::string& input, const char delimiter);
    // Strips padding from the left side of the string
    static void stripPadding(std::string& input, const char padChar);
    // Gets the first element in the provided vector that is not something
    // Returns the excluded element all excluded
    template <typename T>
    static T firstNotOf(const std::vector<T>& vector, const T& excluded) {
        for (const T& elem : vector)
            if (elem != excluded)
                return elem;
        return excluded;
    }
};
