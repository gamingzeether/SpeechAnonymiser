#pragma once

#include "../common_inc.hpp"

#include <vector>
#include <map>
#include <string>
#include <armadillo>
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

  inline static arma::fmat similarityMatrix;
  inline static std::map<std::string, size_t> similarityIndex;
  inline static std::vector<std::string> similarityManifest;

  size_t getClosest(const std::string& fromPhn, const PhonemeSet& to);
  static void loadSimilarityMatrix();
  static std::vector<std::string> splitAndStrip(std::ifstream& fstream);
  static bool isStaticInit() { return similarityMatrix.n_elem > 0; };
  static std::vector<bool> getMask(const PhonemeSet& ps);
};
