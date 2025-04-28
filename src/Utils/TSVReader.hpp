#pragma once

#include "../common_inc.hpp"

#include <fstream>
#include <vector>
#include <cstdint>

class TSVReader {
public:
  struct TSVLine {
    std::string CLIENT_ID; // index 0
    std::string PATH; // index 1
    //std::string SENTENCE_ID; // index 2
    std::string SENTENCE; // index 3
    //std::string SENTENCE_DOMAIN; // index 4
    //std::string UP_VOTES; // index 5
    //std::string DOWN_VOTES; // index 6
    //std::string AGE; // index 7
    //std::string GENDER; // index 8
    //std::string ACCENTS; // index 9
    //std::string VARIANT; // index 10
    //std::string LOCALE; // index 11
    //std::string SEGMENT; // index 12
  };
  struct CompactTSVLine {
    uint64_t* CLIENT_ID; // index 0
    uint32_t PATH; // index 1
    //std::string SENTENCE_ID; // index 2
    std::string SENTENCE; // index 3
    //std::string SENTENCE_DOMAIN; // index 4
    //std::string UP_VOTES; // index 5
    //std::string DOWN_VOTES; // index 6
    //std::string AGE; // index 7
    //std::string GENDER; // index 8
    //std::string ACCENTS; // index 9
    //std::string VARIANT; // index 10
    //std::string LOCALE; // index 11
    //std::string SEGMENT; // index 12
  };

  static TSVLine convert(const CompactTSVLine& compact);
  static CompactTSVLine convert(const TSVLine& compact);

  std::vector<TSVReader::CompactTSVLine> read(const std::string& path);
private:
  std::ifstream reader;
  std::vector<std::string> columns;
};
