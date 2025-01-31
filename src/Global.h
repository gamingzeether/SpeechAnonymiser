#pragma once

#include "common_inc.h"

#include "mutex"
#include "PhonemeSet.h"
#include "Logger.h"

class Global {
public:
	static Global& get() {
		static Global instance = Global();
		return instance;
	};

	const PhonemeSet& phonemeSet() { return ps; };
	// Global lock to prevent multiple fftw functions being executed at the same time eg. planning
	std::unique_lock<std::mutex> fftwLock() { return std::unique_lock<std::mutex>(fftwMutex); };
private:
	Global();

	void initPhonemeSet();

	PhonemeSet ps;
	Logger logger;
	std::mutex fftwMutex;
};
