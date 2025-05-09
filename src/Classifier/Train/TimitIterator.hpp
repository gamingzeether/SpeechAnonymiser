#pragma once

#include "../../common_inc.hpp"

#include <vector>
#include <string>
#include <filesystem>
#include "DatasetIterator.hpp"

class TimitIterator : public DatasetIterator {
public:
  void load(std::filesystem::path path, size_t sr);
  size_t nextClip(Clip& clip, std::vector<Phone>& phones);
  bool good();
  void drop(size_t index);
  void shuffle();
private:
  std::vector<Phone> parsePhones(const std::filesystem::path path);
  std::vector<std::string> paths;
};
