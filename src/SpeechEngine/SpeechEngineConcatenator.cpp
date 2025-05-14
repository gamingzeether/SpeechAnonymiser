#include "SpeechEngineConcatenator.hpp"

#include "../Utils/Global.hpp"

SpeechEngineConcatenator& SpeechEngineConcatenator::configure(std::string file) {
  Global::get().setSpeechEnginePhonemeSet("ConcatSE", true);
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
    desiredFeatures.prev = lastUnit.unit;
  } else {
    desiredFeatures.prev = NULL;
  }
  
  if (desiredFeatures.prev != NULL && desiredFeatures.prev->features.to == frame.phoneme)
    return;

  desiredFeatures.to = frame.phoneme;
  Voicebank& vb = voicebanks[2];
  const Voicebank::Unit& selectUnit = vb.selectUnit(desiredFeatures);
  vb.loadUnit(selectUnit.index);
  playUnit(selectUnit);
}

void SpeechEngineConcatenator::writeBuffer(OUTPUT_TYPE* outputBuffer, unsigned int nFrames) {
  std::unique_lock<std::mutex> lock(stateMutex);
  while (nFrames > 0 && activeUnits.size() > 0) {
    ActiveUnit& au = activeUnits[0];
    
    OUTPUT_TYPE val = au.unit->audio[au.pointer++];
    if (activeUnits.size() == 1) {
      // Continiously loop section
      const size_t loopFadeSamples = 32;
      int sinceLoopTail = au.pointer - (au.unit->audio.size() - loopFadeSamples);
      if (sinceLoopTail > 0) {
        if (au.unit->features.to == G_P_SIL_S) {
          // Don't loop if its just silence
          activeUnits.erase(activeUnits.begin());
        } else {
          // Crossfade into next loop
          double lerpFactor = (double)(sinceLoopTail) / loopFadeSamples;
          size_t newLoopPos = au.unit->consonant + sinceLoopTail;
          val = Util::lerp(val, au.unit->audio[newLoopPos], lerpFactor);
          if (sinceLoopTail == loopFadeSamples) {
            // Reset pointer
            au.pointer = newLoopPos;
          }
        }
      }
    } else {
      // Transition to next unit
      ActiveUnit& nau = activeUnits[1];
      // Start crossfading
      double lerpFactor = (double)(nau.pointer) / nau.unit->overlap;
      val = Util::lerp(val, nau.unit->audio[nau.pointer++], lerpFactor);
      if (nau.pointer >= nau.unit->overlap || au.pointer >= au.unit->audio.size()) {
        // Pop the used unit
        activeUnits.erase(activeUnits.begin());
      }
    }
    assert(abs(val) < 1);
    std::fill_n(outputBuffer, channels, val);
    outputBuffer += channels;
    nFrames--;
  }
  // Fill the remaining buffer with zeros
  std::fill_n(outputBuffer, nFrames * channels, 0);
}

void SpeechEngineConcatenator::_init() {
    G_LG("Type: Concatenator", Logger::INFO);
}

void SpeechEngineConcatenator::playUnit(const Voicebank::Unit& unit) {
  std::unique_lock<std::mutex> lock(stateMutex);

  activeUnits.emplace_back();
  ActiveUnit& aunit = activeUnits.back();
  aunit.unit = &unit;
  aunit.pointer = 0;
}
