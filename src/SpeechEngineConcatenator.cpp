#include "SpeechEngineConcatenator.h"

SpeechEngineConcatenator& SpeechEngineConcatenator::configure(std::string file) {
	_init();
	return *this;
}

void SpeechEngineConcatenator::pushFrame(const SpeechFrame& frame) {

}

void SpeechEngineConcatenator::writeBuffer(OUTPUT_TYPE* outputBuffer, unsigned int nFrames) {

}

void SpeechEngineConcatenator::_init() {

}
