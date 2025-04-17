#pragma once

#include "../common_inc.hpp"

#include "mutex"
#include "Logger.hpp"
#include "PhonemeCollection.hpp"

#define G_LG(...) Global::log(__VA_ARGS__)

#define G_PS_C Global::get().getClassifierPhonemeSet()
#define G_PS_S Global::get().getSpeechEnginePhonemeSet()
#define G_P_SIL Global::get().silencePhone()

#define PHONEME_SET_IPA 0
#define PHONEME_SET_ARPA 1
#define PHONEME_SET_CV 2
#define PHONEME_SET_TIMIT 3

class Global {
public:
	static bool initalized() { return init; };

	static Global& get() {
		static Global instance = Global();
		return instance;
	};

	// Gets the phoneme set with specified id
	const PhonemeSet& getPhonemeSet(size_t id);

	// Checks if classifier phoneme set is set
	bool isClassifierSet();
	// Sets the phoneme set used for classifier
	void setClassifierPhonemeSet(const std::string& name, bool supress = false);
	// Gets the phoneme set used for classifier
	const PhonemeSet& getClassifierPhonemeSet();
	// Gets the name of the phoneme set used for classifier
	std::string getClassifierPhonemeSetName();

	// Checks if speech engine phoneme set is set
	bool isSpeechEngineSet();
	// Sets the phoneme set used for classifier
	void setSpeechEngineSet(const std::string& name);
	// Gets the phoneme set used for speech engine
	const PhonemeSet& getSpeechEnginePhonemeSet();
	// Gets the name of the phoneme set used for speech engine
	std::string getSpeechEnginePhonemeSetName();

	// Sets the phoneme set used for speech engine
	void setSpeechEnginePhonemeSet(const std::string& name, bool supress = false);

	// Returns classifier silence phoneme id
	size_t silencePhone();
	// Global lock to prevent multiple fftw functions being executed at the same time eg. planning
	std::unique_lock<std::mutex> fftwLock() { return std::unique_lock<std::mutex>(fftwMutex); };

	static void log(const std::string& message, int verbosity = 0, int color = Logger::Color::DEFAULT);
private:
	Global();
	inline static bool init = false;
	inline static std::ofstream logFile;

	PhonemeCollection pc;
	Logger logger;
	std::mutex fftwMutex;
	size_t silPhone;

	int classifierPhonemeSetId = -1;
	int speechEnginePhonemeSetId = -1;
};
