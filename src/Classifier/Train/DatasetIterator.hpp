#pragma once

#include "../../common_inc.hpp"

#include <filesystem>
#include <vector>
#include "Clip.hpp"
#include "../../structs.hpp"

class DatasetIterator {
public:
  virtual void load(const std::filesystem::path path, size_t sr) = 0;
  virtual size_t nextClip(Clip& clip, std::vector<Phone>& phones) = 0;
  virtual bool good() = 0;
  virtual void drop(size_t index) = 0;
  virtual void shuffle() = 0;
protected:
  virtual std::vector<Phone> parsePhones(const std::filesystem::path path) = 0;
  size_t pointer;
  size_t sampleRate;
  std::filesystem::path directory;
};
