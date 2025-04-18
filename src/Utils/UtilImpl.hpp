#pragma once
// Implementations for templated functions in Util

#include "Util.hpp"

template <typename T>
T Util::firstNotOf(const std::vector<T>& vector, const T& excluded) {
    for (const T& elem : vector)
        if (elem != excluded)
            return elem;
    return excluded;
}

template <typename... Args>
std::string Util::format(const std::string& format, Args ... args) {
    int strSize = snprintf(NULL, 0, format.c_str(), args...);
    char* cstr = new char[strSize];
    sprintf(cstr, format.c_str(), args...);
    std::string outStr(cstr);
    free(cstr);
    return outStr;
}
