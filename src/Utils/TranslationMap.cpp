#include "TranslationMap.hpp"

#include <fstream>
#include "Global.hpp"

size_t TranslationMap::translate(size_t in) {
  return map[in];
}
  
TranslationMap::TranslationMap(const PhonemeSet& from, const PhonemeSet& to) {
  fromSet = from.id;
  toSet = to.id;

  size_t fromCount = from.invXSampaMap.size();
  map.resize(fromCount);
  for (size_t i = 0; i < fromCount; i++) {
    // Get every X-SAMPA representation in the 'from' set
    std::string fromXSampa = from.xSampa(i);
    // Find the closest neighbor in the 'to' set
    size_t closestIdx = getClosest(fromXSampa, to);
    // Write that down in the map
    map[i] = closestIdx;
  }
}

size_t TranslationMap::getClosest(const std::string& fromPhn, const PhonemeSet& to) {
  if (!isStaticInit())
    loadSimilarityMatrix();
  
  size_t fromIdx = similarityIndex[fromPhn];
  // Get a mask so we know which ones are available
  std::vector<bool> mask = getMask(to);
  // Find the max in similarity matrix and in mask
  float maxSimilarity = -std::numeric_limits<float>::infinity();
  int maxIdx = -1;
  for (size_t i = 0; i < to.size(); i++) {
    if (!mask[i])
      continue;
    size_t toIdx = similarityIndex[to.xSampa(i)];
    float similarity = similarityMatrix(fromIdx, toIdx);
    if (similarity > maxSimilarity) {
      maxIdx = toIdx;
      maxSimilarity = similarity;
    }
  }

  if (maxSimilarity < 1e-8) {
    G_LG(Util::format("Max similarity from phoneme '%s' to set '%s' is zero", fromPhn.c_str(), to.name.c_str()), Logger::WARN);
  }

  return maxIdx;
}

void TranslationMap::loadSimilarityMatrix() {
  // The similarity matrix is a plain text file
  // It is seperated into a grid
  // The top of each column is to "to" phoneme
  // The leftmost element of each row is the "from" phoneme
  // Each cell is 5 characters wide and left padded by spaces
  // Columns are seperated by semicolons and rows by newlines

  // Open the file for reading
  std::ifstream matrixFile("configs/similarity-matrix.txt");
  // For this some reason this check fails even if the rest works
  //if (!matrixFile.is_open())
  //  G_LG("Failed to open similarity matrix", Logger::DEAD);

  std::vector<std::string> phonemes;
  // Load first line as labels
  // The first element should be ignored because it is the column where row labels go
  phonemes = splitAndStrip(matrixFile);
  similarityMatrix = arma::fmat(phonemes.size() - 1, phonemes.size() - 1);
  // Load each row
  for (size_t r = 1; r < phonemes.size(); r++) {
    std::vector<std::string> row = splitAndStrip(matrixFile);
    // Check if the labels are aligned
    if (row.size() != phonemes.size()) {
      G_LG(Util::format("Row %ld has incorrect number of elements", r), Logger::ERRO);
    }
    if (row[0] != phonemes[r]) {
      G_LG(Util::format("Labels do not match: %s and %s (line %ld)", row[0].c_str(), phonemes[r].c_str(), r), Logger::ERRO);
    }
    // Load the values into the similarity matrix
    for (size_t c = 1; c < row.size(); c++) {
      std::string elem = row[c];
      float val;
      try {
        val = std::stof(elem);
      } catch (...) {
        G_LG(Util::format("Failed to parse %s as a float", elem.c_str()), Logger::ERRO);
      }
      similarityMatrix(r - 1, c - 1) = val;
    }
  }

  matrixFile.close();
}

std::vector<std::string> TranslationMap::splitAndStrip(std::ifstream& fstream) {
  std::string line;
  std::getline(fstream, line);

  std::vector<std::string> row;
  row = Util::split(line, ';');
  for (std::string& elem : row)
    Util::stripPadding(elem, ' ');
  return row;
}

std::vector<bool> TranslationMap::getMask(const PhonemeSet& ps) {
  size_t size = ps.size();
  std::vector<bool> mask(size, false);
  for (size_t i = 0; i < size; i++) {
    size_t idx = similarityIndex[ps.xSampa(i)];
    mask[idx] = true;
  }
  return mask;
}
