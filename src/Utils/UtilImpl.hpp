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
  int strSize = snprintf(NULL, 0, format.c_str(), args...) + 1;
  char* cstr = new char[strSize];
  snprintf(cstr, strSize, format.c_str(), args...);
  std::string outStr(cstr);
  delete[] cstr;
  return outStr;
}

template <typename Type, typename AmountType>
Type Util::lerp(Type start, Type end, AmountType t) {
  return start + (end - start) * t;
}

template <typename Type>
bool Util::fcmp(Type a, Type b, Type eps) {
  Type diff = std::abs(a - b);
  return diff <= eps;
}

template <typename VecType>
std::string Util::vecToString(const std::vector<VecType>& vec) {
  std::string line = "[";
  for (size_t i = 0; i < vec.size(); i++) {
    line += std::to_string(vec[i]);
    if (i < vec.size() - 1) {
      line += ", ";
    } else {
      line += "]";
    }
  }
  return line;
}
