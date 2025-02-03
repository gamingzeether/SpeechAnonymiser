#pragma once

#include "SpeechEngine.hpp"
#include "Voicebank.hpp"

// Concatenator based speech synthesiser

class SpeechEngineConcatenator : public SpeechEngine {
public:
	SpeechEngineConcatenator() {};

	virtual void pushFrame(const SpeechFrame& frame);
	virtual void writeBuffer(OUTPUT_TYPE* outputBuffer, unsigned int nFrames);

	virtual SpeechEngineConcatenator& configure(std::string file);
protected:
	struct ActiveUnit {
		const Voicebank::Unit* unit;
		size_t pointer;
	};

	virtual void _init();
	virtual void playUnit(const Voicebank::Unit& unit);

	std::vector<Voicebank> voicebanks;
	std::vector<ActiveUnit> activeUnits;

	size_t currentPhoneme = 0;
};
