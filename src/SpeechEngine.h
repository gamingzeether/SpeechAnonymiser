#pragma once

#include "common_inc.h"

#include <chrono>
#include <random>
#include <unordered_map>
#include <string>
#include "structs.h"
#include "define.h"
#include "Logger.h"
#include "JSONHelper.h"

/* Defines a base speech engine
 * Do not use this
 */

class SpeechEngine {
public:
	// Pass frame containing phoneme to engine
	virtual void pushFrame(const SpeechFrame& frame) abstract;
	// Write audio into specified buffer
	virtual void writeBuffer(OUTPUT_TYPE* outputBuffer, unsigned int nFrames) abstract;

	virtual SpeechEngine& setSampleRate(int sr) { 
		sampleRate = sr;
		return *this;
	}

	virtual SpeechEngine& setChannels(int ch) {
		channels = ch;
		return *this;
	}

	// Call this last
	virtual SpeechEngine& configure(std::string file) abstract;
protected:
	static float _random() {
		std::uniform_real_distribution<float> randomDist = std::uniform_real_distribution<float>(-1, 1);
		return randomDist(randomEngine);
	};
	virtual void _init() abstract;

	Logger logger;

	int sampleRate = 16000;
	int channels = 1;

	inline static std::default_random_engine randomEngine = std::default_random_engine((unsigned int)std::chrono::system_clock::now().time_since_epoch().count());
};
