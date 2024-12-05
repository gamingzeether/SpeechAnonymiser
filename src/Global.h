#pragma once

#include "common_inc.h"

#include "PhonemeSet.h"
#include "Logger.h"

class Global {
public:
	static Global get() {
		static Global instance = Global();
		return instance;
	};

	const PhonemeSet& phonemeSet() { return ps; };
private:
	Global();

	void initPhonemeSet();

	PhonemeSet ps;
	Logger logger;
};
