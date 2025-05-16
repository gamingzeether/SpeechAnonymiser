#pragma once

#include "../common_inc.hpp"

#include <QObject>
#include <QSlider>

#include "../Classifier/Train/Dataset.hpp"
#include "../SpeechEngine/SpeechEngine.hpp"
#include "../GUI/Spectrogram.hpp"
#include "../include_la.hpp"

class InteractiveDebugger : public QObject {
public:
  InteractiveDebugger(size_t sampleRate, std::string dataPath, AudioContainer& container) : 
      sampleRate(sampleRate),
      ds(sampleRate, dataPath),
      audioContainer(&container) {
    ds.setSubtype(Subtype::TEST);
  };
  void run();
private:
  InteractiveDebugger();
  void initWindow();
  void loadCubeSpectrogram(const CPU_CUBE_TYPE& data, size_t col);
  void hidePhoneMarkers(size_t start = 0);
  void placePhoneMarkers(const CPU_CUBE_TYPE& labels, size_t col);
  void findAndLoadClip();
  void playSpeech(size_t phoneme);
  void playSpeechSequence();
  void addSpeechEngine();

  QWidget* qtWindow;
  Dataset ds;
  AudioContainer* audioContainer;
  std::unique_ptr<SpeechEngine> speechEngine;
  size_t sampleRate;
  Clip currentClip;

  Spectrogram* spectrogram;
  QSlider* clipSlider;

  size_t selectedPhoneme = 0;
  size_t selectedPhonemeSpeech = 0;

  bool sliderSelected = false;
  std::vector<QLabel*> clipPhoneMarkers;
};
