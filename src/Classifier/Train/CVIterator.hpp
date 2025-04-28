#pragma once

#include "../../common_inc.hpp"

#include <vector>
#include <cstdint>
#include "DatasetIterator.hpp"
#include "../../Utils/TSVReader.hpp"

class CVIterator : public DatasetIterator {
public:
  void load(std::filesystem::path path, size_t sr);
  size_t nextClip(Clip& clip, TSVReader::TSVLine& tsv);
  size_t nextClip(Clip& clip, std::vector<Phone>& phones);
  bool good();
  void drop(size_t index);
  void shuffle();

  void filter(std::string client);
private:
  std::vector<Phone> parsePhones(const std::filesystem::path path);

  std::vector<std::string> columns;
  std::vector<TSVReader::CompactTSVLine> lines;
  std::string appliedFilter = "";
};
