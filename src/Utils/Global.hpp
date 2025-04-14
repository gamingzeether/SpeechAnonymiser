#pragma once

#include "../common_inc.hpp"

#include "mutex"
#include "Logger.hpp"
#include "PhonemeSet.hpp"

#define G_LG(...) Global::get().log(__VA_ARGS__)
#define G_PS Global::get().phonemeSet()

class Global {
public:
	static Global& get() {
		static Global instance = Global();
		return instance;
	};

	const PhonemeSet& phonemeSet() { return ps; };
	size_t silencePhone();
	// Global lock to prevent multiple fftw functions being executed at the same time eg. planning
	std::unique_lock<std::mutex> fftwLock() { return std::unique_lock<std::mutex>(fftwMutex); };

	void log(const std::string& message, int verbosity = 0, int color = Logger::Color::DEFAULT);
private:
	Global();

	void initPhonemeSet();

	PhonemeSet ps;
	Logger logger;
	std::mutex fftwMutex;
	size_t silPhone;
};
