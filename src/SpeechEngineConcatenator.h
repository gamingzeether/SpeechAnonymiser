#include "SpeechEngine.h"

// Concatenator based speech synthesiser

class SpeechEngineConcatenator : public SpeechEngine {
public:
	SpeechEngineConcatenator() {};

	virtual void pushFrame(const SpeechFrame& frame);
	virtual void writeBuffer(OUTPUT_TYPE* outputBuffer, unsigned int nFrames);

	virtual SpeechEngineConcatenator& configure(std::string file);
protected:
	virtual void _init();
};
