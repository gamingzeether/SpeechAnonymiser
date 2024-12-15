#include "PhonemeSet.h"

#include "Util.h"

size_t PhonemeSet::fromString(const std::string& str) const {
	return fromString(Util::utf8_to_utf16(str));
}

PhonemeSet& PhonemeSet::addString(const std::string& str, const std::string& xSampa) {
	return addString(Util::utf8_to_utf16(str), xSampa);
}

size_t PhonemeSet::fromString(const std::wstring& str) const {
	size_t hash = Util::customHasher(str);
	return stringMap.at(hash);
}

PhonemeSet& PhonemeSet::addString(const std::wstring& str, const std::string& xSampa) {
	size_t hash = Util::customHasher(str);
	if (stringMap.find(hash) != stringMap.end()) {
		if (xSampaMap[xSampa] == stringMap[hash])
			return *this;
		throw("Hash collision");
	}
	stringMap[hash] = getOrNew(xSampa);
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
