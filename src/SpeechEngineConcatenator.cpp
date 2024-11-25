#include "SpeechEngineConcatenator.h"

SpeechEngineConcatenator& SpeechEngineConcatenator::configure(std::string file) {
	for (auto& subdir : {"B3_Soft/", "B4_Power/", "D#4_Natural/", "G3_Soft/", "G4_Natural/"}) {
		Voicebank vb;
		vb.targetSamplerate(sampleRate)
			.open(file + subdir);
		voicebanks.push_back(std::move(vb));
	}
	_init();
	return *this;
}

void SpeechEngineConcatenator::pushFrame(const SpeechFrame& frame) {

}

void SpeechEngineConcatenator::writeBuffer(OUTPUT_TYPE* outputBuffer, unsigned int nFrames) {

}

void SpeechEngineConcatenator::_init() {

}
