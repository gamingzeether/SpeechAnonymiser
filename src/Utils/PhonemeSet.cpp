#include "PhonemeSet.hpp"

#include "Util.hpp"

size_t PhonemeSet::fromString(const std::string& str) const {
	return fromString(Util::utf8_to_utf16(str));
}

PhonemeSet& PhonemeSet::addString(const std::string& str, const std::string& xSampa) {
	return addString(Util::utf8_to_utf16(str), xSampa);
}

size_t PhonemeSet::fromString(const std::wstring& str) const {
	return stringMap.at(str);
}

PhonemeSet& PhonemeSet::addString(const std::wstring& str, const std::string& xSampa) {
	if (stringMap.find(str) != stringMap.end()) {
		if (xSampaMap[xSampa] != stringMap[str])
			std::wprintf(L"Collision: %s:%s", str.c_str(), Util::utf8_to_utf16(xSampa).c_str());
	}
	stringMap[str] = getOrNew(xSampa);
	return *this;
}

std::string PhonemeSet::xSampa(size_t phoneme) const {
	return invXSampaMap.at(phoneme);
}

size_t PhonemeSet::xSampaIndex(const std::string& str) const {
	return xSampaMap.at(str);
}

bool PhonemeSet::xSampaExists(const std::string& str) const {
	return xSampaMap.find(str) != xSampaMap.end();
}

size_t PhonemeSet::getOrNew(const std::string& xSampa) {
	auto iter = xSampaMap.find(xSampa);
	if (iter == xSampaMap.end()) {
		invXSampaMap[_counter] = xSampa;
		xSampaMap[xSampa] = _counter;
		return _counter++;
	}
	return iter->second;
}
