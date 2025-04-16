#include "PhonemeSet.hpp"

#include "Util.hpp"
#include "Global.hpp"

size_t PhonemeSet::fromString(const std::string& str) const {
	return fromString(Util::str2wstr(str));
}

PhonemeSet& PhonemeSet::addString(const std::string& str, const std::string& xSampa) {
	return addString(Util::str2wstr(str), xSampa);
}

size_t PhonemeSet::fromString(const std::wstring& str) const {
	if (!has(stringMap, str)) {
		G_LG(std::format("Set {} does not have phoneme '{}'", this->name, Util::wstr2str(str)), Logger::ERRO);
		return invalidPhoneme;
	}
	return stringMap.at(str);
}

PhonemeSet& PhonemeSet::addString(const std::wstring& str, const std::string& xSampa) {
	if (stringMap.find(str) != stringMap.end()) {
		if (xSampaMap[xSampa] != stringMap[str])
			G_LG(std::format("Collision: {} / {}", Util::wstr2str(str), xSampa), Logger::ERRO);
	}
	stringMap[str] = getOrNew(xSampa);
	return *this;
}

std::string PhonemeSet::xSampa(size_t phoneme) const {
	if (phoneme >= invXSampaMap.size()) {
		G_LG(std::format("Set {} does not have phoneme ID {}", this->name, phoneme), Logger::DEAD);
		return invalidString;
	}
	return invXSampaMap.at(phoneme);
}

size_t PhonemeSet::xSampaIndex(const std::string& str) const {
	if (!has(xSampaMap, str)) {
		G_LG(std::format("Set {} does not have X-SAMPA phoneme '{}'", this->name, str), Logger::ERRO);
		return invalidPhoneme;
	}
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
