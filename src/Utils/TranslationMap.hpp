#pragma once

#include "../common_inc.hpp"

#include <vector>
#include <map>
#include <string>
#include "PhonemeSet.hpp"

// Each PhonemeSet's xSampaMap has different indices
// This translates one to another (or its closest neighbor)
class TranslationMap {
public:
  size_t translate(size_t in);

  TranslationMap(const PhonemeSet& from, const PhonemeSet& to);
private:
  int fromSet;
  int toSet;
  std::vector<size_t> map;

  double distance(const std::string& from, const std::string& to);
};
