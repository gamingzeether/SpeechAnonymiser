#include "SpeechEngineConcatenator.h"

SpeechEngineConcatenator& SpeechEngineConcatenator::configure(std::string file) {
	std::vector<std::string> subdirs = { "B3_Soft/", "B4_Power/", "D#4_Natural/", "G3_Soft/", "G4_Natural/" };
	for (std::string& subdir : subdirs) {
		Voicebank vb;
		std::string shortName = subdir.substr(0, subdir.find('_'));
		vb.targetSamplerate(sampleRate)
			.setShort(shortName)
			.open(file + subdir);
		voicebanks.push_back(std::move(vb));
	}
	_init();
	return *this;
}

void SpeechEngineConcatenator::pushFrame(const SpeechFrame& frame) {
	Voicebank::DesiredFeatures desiredFeatures;
	if (activeUnits.size() > 0) {
		const ActiveUnit& lastUnit = activeUnits.back();
		desiredFeatures.from = lastUnit.unit;
		return;
	} else {
		desiredFeatures.from = NULL;
	}
	
	if (!desiredFeatures.from || desiredFeatures.from->features.to == frame.phoneme) {
		return;
	}

	desiredFeatures.to = frame.phoneme;
	Voicebank& vb = voicebanks[2];
	const Voicebank::Unit& selectUnit = vb.selectUnit(desiredFeatures);
	vb.loadUnit(selectUnit.index);
	playUnit(selectUnit);
}

void SpeechEngineConcatenator::writeBuffer(OUTPUT_TYPE* outputBuffer, unsigned int nFrames) {
	for (int i = 0; i < nFrames * channels; i++) {
		outputBuffer[i] = 0;
	}
	std::unique_lock<std::mutex> lock(stateMutex);
	for (ActiveUnit& aunit : activeUnits) {
		const Voicebank::Unit& unit = *aunit.unit;
		size_t unitSize = unit.audio.size();
		unsigned int samples = std::min(nFrames, (unsigned int)(unitSize - aunit.pointer));
		OUTPUT_TYPE* outputTmp = outputBuffer;
		for (int i = 0; i < samples; i++) {
			float data = unit.audio[aunit.pointer + i] * volume;
			for (int j = 0; j < channels; j++) {
				*(outputTmp++) += data;
			}
		}
		aunit.pointer += samples;
	}
	if (activeUnits.size() > 0) {
		ActiveUnit& front = activeUnits.front();
		if (front.pointer >= front.unit->audio.size()) {
			for (int i = 1; i < activeUnits.size(); i++) {
				activeUnits[i - 1] = std::move(activeUnits[i]);
			}
			activeUnits.pop_back();
		}
	}
}

void SpeechEngineConcatenator::_init() {

}

void SpeechEngineConcatenator::playUnit(const Voicebank::Unit& unit) {
	std::unique_lock<std::mutex> lock(stateMutex);

	activeUnits.push_back(ActiveUnit());
	ActiveUnit& aunit = activeUnits.back();
	aunit.unit = &unit;
	aunit.pointer = 0;
}
