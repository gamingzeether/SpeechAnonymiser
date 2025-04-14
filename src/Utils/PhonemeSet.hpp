#pragma once

#include "../common_inc.hpp"

#include <string>
#include <map>

// Access through Global::get().phonemeSet()
// Or the macro G_PS defined in Global.hpp
class PhonemeSet {
public:
	struct Phoneme {
		size_t index;
		std::string symbol;
		std::string diacritic;
	};

	size_t fromString(const std::string& str) const;
	PhonemeSet& addString(const std::string& str, const std::string& xSampa);
	size_t fromString(const std::wstring& str) const;
	PhonemeSet& addString(const std::wstring& str, const std::string& xSampa);

	std::string xSampa(size_t phoneme) const;
	size_t xSampaIndex(const std::string& str) const;
	bool xSampaExists(const std::string& str) const;
	size_t size() const { return _counter; };
private:
	std::map<std::string, size_t> xSampaMap;
	std::map<size_t, std::string> invXSampaMap;
	std::map<std::wstring, size_t> stringMap;

	size_t _counter = 0;

	PhonemeSet() {};

	size_t getOrNew(const std::string& xSampa);

	friend class Global;
};
