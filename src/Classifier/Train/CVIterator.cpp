#include "CVIterator.hpp"

#include <string>
#include <random>
#include <chrono>
#include <fstream>
#include "../../Utils/Global.hpp"
#include "../../structs.hpp"

void CVIterator::load(std::filesystem::path path, size_t sr) {
  sampleRate = sr;
  TSVReader reader;
  lines = reader.read(path);
  directory = path.parent_path();
}

size_t CVIterator::nextClip(Clip& clip, TSVReader::TSVLine& tsv) {
  size_t i = pointer++;
  tsv = TSVReader::convert(lines[i]);
  const std::string clipPath = Util::format("%s/clips/%s",
      directory.c_str(), tsv.PATH.c_str());
  clip.setClipPath(clipPath);
  return i;
}

size_t CVIterator::nextClip(Clip& clip, std::vector<Phone>& phones) {
  TSVReader::TSVLine tsv;
  size_t index = nextClip(clip, tsv);
  const std::string transcriptionPath = Util::format("%s/transcript/%s/%s.TextGrid",
      directory.c_str(), tsv.CLIENT_ID.c_str(), tsv.PATH.substr(0, tsv.PATH.length() - 4).c_str());
  phones = parsePhones(transcriptionPath);
  return index;
}

bool CVIterator::good() {
  return pointer < lines.size();
}

void CVIterator::drop(size_t index) {
  lines[index] = std::move(lines.back());
  lines.pop_back();
}

void CVIterator::shuffle() {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine random(seed);
  std::shuffle(lines.begin(), lines.end(), random);
  pointer = 0;
}

void CVIterator::filter(std::string client) {
  if (client == appliedFilter) {
    return;
  } else if (appliedFilter != "") {
    G_LG("Tried to change filter when one was already applied", Logger::ERRO);
  }
  TSVReader::TSVLine expanded;
  size_t index = lines.size();
  while (index > 0) {
    index--;
    expanded = TSVReader::convert(lines[index]);
    if (expanded.CLIENT_ID != client)
      drop(index);
  }
  appliedFilter = client;
}

std::vector<Phone> CVIterator::parsePhones(const std::filesystem::path path) {
  std::ifstream reader;
  reader.open(path);
  if (!reader.is_open()) {
    G_LG(Util::format("Failed to open file at %s", path.c_str()), Logger::ERRO);
    return {};
  }

  std::string line;
  // Seek to a specific line where phones start
  while (reader.good()) {
    std::getline(reader, line);
    if (line == "        name = \"phones\" ") {
      for (int i = 0; i < 2; i++) {
        std::getline(reader, line);
      }
      break;
    }
  }

  // Get next line (number of phones)
  std::getline(reader, line);
  std::string sizeString = line.substr(line.find_last_of(' ', line.length() - 2));
  int size = std::stoi(sizeString);
  std::vector<Phone> phones(size);

  // Load each phoneme
  std::string interval, xmin, xmax, text;
  for (int i = 0; i < size; i++) {
    std::getline(reader, interval);
    std::getline(reader, xmin);
    std::getline(reader, xmax);
    std::getline(reader, text);

    xmin = xmin.substr(xmin.find_last_of(' ', xmin.length() - 2));
    xmax = xmax.substr(xmax.find_last_of(' ', xmax.length() - 2));
    size_t textStart = text.find_last_of('\"', text.length() - 3) + 1;
    text = text.substr(textStart, text.length() - textStart - 2);

    Phone p = Phone();
    p.min = std::stod(xmin);
    p.max = std::stod(xmax);
    p.minIdx = sampleRate * p.min;
    p.maxIdx = sampleRate * p.max;

    p.phonetic = G_PS_C.fromString(text);
    phones[i] = p;
  }

  reader.close();

  return phones;
}
