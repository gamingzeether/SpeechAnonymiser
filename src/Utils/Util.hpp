#pragma once

#include "../common_inc.hpp"

#include <vector>
#include <string>
#include <locale>
#include <codecvt>

#define CS c_str()

class Util {
public:
    // As the name describes
    static std::wstring str2wstr(const std::string& utf8);
    // As the name describes
    static std::string wstr2str(const std::wstring& utf16);
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
    static T firstNotOf(const std::vector<T>& vector, const T& excluded);
    // Checks if a contains b
    static bool contains(const std::string& a, const std::string& b);
    // Formats a string similar to c++20's std::format
    template <typename... Args>
    static std::string format(const std::string& format, Args ... args);
    // Performs a linear interpolation
    template <typename Type, typename AmountType>
    static Type lerp(Type start, Type end, AmountType t);
    // Checks if a is close to b
    template <typename Type>
    static bool fcmp(Type a, Type b, Type eps);
private:
    inline static std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t> cvt;
};

#include "UtilImpl.hpp"
