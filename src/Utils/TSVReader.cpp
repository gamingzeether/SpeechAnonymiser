#include "TSVReader.hpp"

#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include "Global.hpp"
#include "Util.hpp"
#include "../structs.hpp"

TSVReader::TSVLine TSVReader::convert(const TSVReader::CompactTSVLine& compact) {
  TSVReader::TSVLine expanded;
  std::string idString = "";
  for (int i = 0; i < 8; i++) {
    idString += Util::format("%016llx", compact.CLIENT_ID[i]);
  }
  expanded.CLIENT_ID = idString;
  expanded.PATH = Util::format("common_voice_en_%u.mp3", compact.PATH);
  expanded.SENTENCE = compact.SENTENCE;
  return expanded;
}

TSVReader::CompactTSVLine TSVReader::convert(const TSVReader::TSVLine& expanded) {
  TSVReader::CompactTSVLine compact;
  compact.CLIENT_ID = new uint64_t[8];
  for (int i = 0; i < 8; i++) {
    std::string substr = expanded.CLIENT_ID.substr(i * 16, 16);
    compact.CLIENT_ID[i] = std::stoull(substr, 0, 16);
  }
  std::string pathNumber = expanded.PATH.substr(16);
  pathNumber = pathNumber.substr(0, pathNumber.size() - 4);
  compact.PATH = std::stoi(pathNumber);
  compact.SENTENCE = expanded.SENTENCE;
  return compact;
}

std::vector<TSVReader::CompactTSVLine> TSVReader::read(const std::string& path) {
  std::ifstream reader;
  reader.open(path);
  G_LG(Util::format("Loading TSV: %s", path.c_str()), Logger::INFO);
  if (!reader.is_open())
    G_LG(Util::format("Failed to open file at %s", path.c_str()), Logger::DEAD);

  std::string line;
  std::getline(reader, line);
  columns = Util::split(line, '\t');

  std::vector<CompactTSVLine> lines;
  while (std::getline(reader, line)) {
    std::vector<std::string> elems = Util::split(line, '\t');
    if (elems.size() != columns.size())
      G_LG(Util::format("Failed to parse line in file '%s': %s", path.c_str(), line.c_str()), Logger::WARN);
    
    TSVLine parsedLine;
    for (size_t i = 0; i < columns.size(); i++) {
      switch (i) {
        case 0:
          parsedLine.CLIENT_ID = elems[i];
          break;
        case 1:
          parsedLine.PATH = elems[i];
          break;
        case 3:
          parsedLine.SENTENCE = elems[i];
          break;
      }
    }
    lines.push_back(TSVReader::convert(parsedLine));
  }
  G_LG(Util::format("Loaded %zu lines from '%s'", lines.size(), path.c_str()), Logger::INFO);
  reader.close();
  return lines;
}
