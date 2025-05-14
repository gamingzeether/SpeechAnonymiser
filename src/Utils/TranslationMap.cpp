#include "TranslationMap.hpp"

#include <fstream>
#include <utility>
#include "Global.hpp"

size_t TranslationMap::translate(size_t in) {
  return map[in];
}
  
TranslationMap::TranslationMap(const PhonemeSet& from, const PhonemeSet& to) {
  fromSet = from.id;
  toSet = to.id;

  size_t fromCount = from.size();
  size_t toCount = to.size();

  map.resize(fromCount);
  for (size_t i = 0; i < fromCount; i++) {
    double min = 9e99;
    size_t minIdx = 0;
    std::string fromStr = from.xSampa(i);
    for (size_t j = 0; j < toCount; j++) {
      double dist = distance(fromStr, to.xSampa(j));
      if (dist < min) {
        min = dist;
        minIdx = j;
      }
    }
    // Warn if distance is greater than some threshold
    const double warnDist = 2;
    if (min > warnDist)
      G_LG(Util::format("The distance between %s and its nearest neighbor in %s is greater than %lf",
          fromStr.c_str(), to.getName().c_str(), warnDist), Logger::WARN);
    map[i] = minIdx;
  }
}

// Estimate distance between two phonemes with Levenshtein distance
// https://en.wikipedia.org/wiki/Levenshtein_distance#Iterative_with_two_matrix_rows
double TranslationMap::distance(const std::string& s, const std::string& t) {
  const double DEL_COST = 1;
  const double INS_COST = 1;
  const double SUB_COST = 1;

  size_t m = s.size();
  size_t n = t.size();
  // create two work vectors of integer distances
  std::vector<double> v0(n + 1), v1(n + 1);

  // initialize v0 (the previous row of distances)
  // this row is A[0][i]: edit distance from an empty s to t;
  // that distance is the number of characters to append to  s to make t.
  for (size_t i = 0; i <= n; i++) {
    v0[i] = i;
  }
  
  for (size_t i = 0; i < m; i++) {
    // calculate v1 (current row distances) from the previous row v0

    // first element of v1 is A[i + 1][0]
    //   edit distance is delete (i + 1) chars from s to match empty t
    v1[0] = i + 1;

    // use formula to fill in the rest of the row
    for (size_t j = 0; j < n; j++) {
      // calculating costs for A[i + 1][j + 1]
      double deletionCost = v0[j + 1] + DEL_COST;
      double insertionCost = v1[j] + INS_COST;
      double substitutionCost = (s[i] == t[j]) ? v0[j] : v0[j] + SUB_COST;

      v1[j + 1] = std::min(deletionCost, std::min(insertionCost, substitutionCost));

      // copy v1 (current row) to v0 (previous row) for next iteration
      // since data in v1 is always invalidated, a swap without copy could be more efficient
      std::swap(v0, v1);
    }
  }
  // after the last swap, the results of v1 are now in v0
  return v0[n];
}
