#pragma once

#include "../common_inc.hpp"

#include <string>
#include <map>

class PhonemeSet {
public:
  struct Phoneme {
    size_t index;
    std::string symbol;
    //std::string diacritic;
  };

  size_t fromString(const std::string& str) const;
  PhonemeSet& addString(const std::string& str, const std::string& xSampa);
  size_t fromString(const std::wstring& str) const;
  PhonemeSet& addString(const std::wstring& str, const std::string& xSampa);

  std::string xSampa(size_t phoneme) const;
  size_t xSampaIndex(const std::string& str) const;
  bool xSampaExists(const std::string& str) const;
  size_t size() const { return _counter; };

  std::string getName() const { return name; };
  size_t getId() const { return id; };

  inline static const size_t invalidPhoneme = -1;
  inline static const std::string invalidString = "N/A";
  inline static const std::wstring invalidWString = L"N/A";
private:
  std::map<std::string, size_t> xSampaMap;
  std::map<size_t, std::string> invXSampaMap;
  std::map<std::wstring, size_t> stringMap;

  size_t _counter = 0;

  std::string name;
  int id;

  PhonemeSet(std::string name, size_t id) : name(name), id(id) {};

  size_t getOrNew(const std::string& xSampa);

  template <typename Map, typename Val>
  bool has(const Map& map, const Val& val) const {
    return (map.find(val) != map.end());
  }

  friend class PhonemeCollection;
  friend class TranslationMap;
};
