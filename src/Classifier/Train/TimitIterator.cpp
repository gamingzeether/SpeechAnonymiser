#include "TimitIterator.hpp"

#include <random>
#include <chrono>
#include <fstream>
#include "../../Utils/Global.hpp"

void TimitIterator::load(std::filesystem::path path, size_t sr) {
  sampleRate = sr;
  directory = path;
  size_t rootPathLen = directory.string().length();
  std::filesystem::recursive_directory_iterator iter(path);
  for (const auto& item : iter) {
    if (item.is_regular_file()) {
      auto path = item.path();
      if (path.extension() == ".PHN") {
        std::string pathStr = path.string();
        std::string appendPath = pathStr.substr(rootPathLen, pathStr.length() - rootPathLen - 4);
        paths.push_back(appendPath);
      }
    }
  }
}

size_t TimitIterator::nextClip(Clip& clip, std::vector<Phone>& phones) {
  size_t i = pointer++;
  
  const std::string& fname = directory.string() + paths[i];

  clip.setClipPath(fname + "_.wav");
  phones = parsePhones(fname + ".PHN");
  return i;
}

bool TimitIterator::good() {
  return pointer < paths.size();
}

void TimitIterator::drop(size_t index) {
  paths[index] = std::move(paths.back());
  paths.pop_back();
}

void TimitIterator::shuffle() {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine random(seed);
  std::shuffle(paths.begin(), paths.end(), random);
  pointer = 0;
}

std::vector<Phone> TimitIterator::parsePhones(const std::filesystem::path path) {
  std::ifstream phonemeReader;
  phonemeReader.open(path);
  if (!phonemeReader.is_open()) {
    G_LG(Util::format("Failed to open file at %s\n", path.c_str()), Logger::ERRO);
    return {};
  }

  std::string line;
  std::vector<Phone> tempPhones;
  while (true) {
    std::getline(phonemeReader, line);
    if (line == "")
      break;

    size_t s1 = line.find(' ');
    size_t s2 = line.find(' ', s1 + 1);

    std::string i1 = line.substr(0, s1).c_str();
    std::string i2 = line.substr(s1 + 1, s2 - s1 - 1).c_str();
    std::string i3 = line.substr(s2 + 1).c_str();

    Phone p = Phone();
    p.minIdx = std::stoull(i1);
    p.min = p.minIdx / 16000.0;
    p.maxIdx = std::stoull(i2);
    p.max = p.maxIdx / 16000.0;
    p.phonetic = G_PS_C.fromString(i3);

    tempPhones.push_back(std::move(p));
  };

  phonemeReader.close();
  return tempPhones;
}
