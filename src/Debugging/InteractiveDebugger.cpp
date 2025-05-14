#include "InteractiveDebugger.hpp"

#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QPushButton>
#include <QTimer>

#include "../include_mlpack.hpp"

#include "../SpeechEngine/SpeechEngineConcatenator.hpp"
#include "../Utils/Global.hpp"
#include "../Utils/TranslationMap.hpp"

#define CLIP_LENGTH 8
#define N_SLICES (CLIP_LENGTH * (sampleRate / FFT_FRAME_SPACING))

void InteractiveDebugger::run() {
  addSpeechEngine();
  // Dummy vars for QApplication
  int argC = 0;
  char** argV;
  QApplication qtApp(argC, argV);
  qtWindow = new QWidget();
  initWindow();
  qtApp.exec();
}

void InteractiveDebugger::initWindow() {
  qtWindow->resize(1620, 240);
  qtWindow->show();
  qtWindow->setWindowTitle("Debugger");

  // Add phoneme selection dropdown
  QComboBox* phonemeDropdown = new QComboBox(qtWindow);
  for (size_t i = 0; i < G_PS_C.size(); i++) {
    phonemeDropdown->addItem(G_PS_C.xSampa(i).c_str());
  }
  phonemeDropdown->move(10, 10);
  phonemeDropdown->show();
  QObject::connect(phonemeDropdown, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [&](int index){
    if (index < 0)
      return;
    
    selectedPhoneme = index;
  });

  // Add find clip button
  QPushButton* findClip = new QPushButton(qtWindow);
  findClip->setText("Find clip");
  findClip->move(100, 10);
  findClip->show();
  QObject::connect(findClip, &QPushButton::clicked, this, [&]{
    findAndLoadClip();
  });

  // Add speech phoneme selection dropdown
  QComboBox* phonemeDropdownSpeech = new QComboBox(qtWindow);
  for (size_t i = 0; i < G_PS_S.size(); i++) {
    phonemeDropdownSpeech->addItem(G_PS_S.xSampa(i).c_str());
  }
  phonemeDropdownSpeech->move(200, 10);
  phonemeDropdownSpeech->show();
  QObject::connect(phonemeDropdownSpeech, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [&](int index){
    if (index < 0)
      return;
    
    selectedPhonemeSpeech = index;
  });

  // Add speech engine say button
  QPushButton* speechSay = new QPushButton(qtWindow);
  speechSay->setText("Say");
  speechSay->move(300, 10);
  speechSay->show();
  QObject::connect(speechSay, &QPushButton::clicked, this, [&]{
    playSpeech(selectedPhonemeSpeech);
    spectrogram->clearSpectrogram();
    hidePhoneMarkers();
  });

  // Add speech engine sing button
  QPushButton* speechSing = new QPushButton(qtWindow);
  speechSing->setText("Sing");
  speechSing->move(400, 10);
  speechSing->show();
  QObject::connect(speechSing, &QPushButton::clicked, this, [&]{
    playSpeechSequence();
    spectrogram->clearSpectrogram();
    hidePhoneMarkers();
  });

  // Add spectrogram
  spectrogram = new Spectrogram(qtWindow, N_SLICES);
  spectrogram->move(10, 110);
  spectrogram->show();

  // Add clip slider
  clipSlider = new QSlider(qtWindow);
  int sliderWidth = spectrogram->getSliceX(N_SLICES) - spectrogram->getSliceX(0);
  clipSlider->resize(sliderWidth, 30);
  clipSlider->setMinimum(0);
  clipSlider->setMaximum(CLIP_LENGTH * 1000);
  clipSlider->setOrientation(Qt::Orientation::Horizontal);
  clipSlider->move(10, 80);
  clipSlider->show();
  // Add event for changing seek position
  QObject::connect(clipSlider, &QSlider::sliderPressed, this, [&]() {
    sliderSelected = true;
  });
  QObject::connect(clipSlider, &QSlider::sliderReleased, this, [&]() {
    double time = (double)clipSlider->value() / 1000;
    std::unique_lock lock(audioContainer->mtx);
    size_t position = std::min(audioContainer->audio.size(), (size_t)(time * sampleRate));
    audioContainer->pointer = position;
    sliderSelected = false;
  });

  // Pause toggle
  QCheckBox* pauseToggle = new QCheckBox(qtWindow);
  pauseToggle->move(10, 40);
  pauseToggle->setText("Pause");
  QObject::connect(pauseToggle, &QCheckBox::stateChanged, this, [&](int state){
    audioContainer->pause = (state == 2);
  });
  pauseToggle->show();

  // Timer to move slider with audio
  QTimer* sliderTimer = new QTimer(qtWindow);
  sliderTimer->start(20);
  QObject::connect(sliderTimer, &QTimer::timeout, this, [&](){
    if (sliderSelected)
      return;
    double time = (double)audioContainer->pointer / sampleRate;
    clipSlider->setValue(time * 1000);
  });
}

void InteractiveDebugger::loadCubeSpectrogram(const CPU_CUBE_TYPE& data, size_t col) {
  spectrogram->clearSpectrogram();
  std::array<float, FRAME_SIZE> frame;
  for (size_t s = 0; s < data.n_slices; s++) {
    CUBE_TYPE inputTemp;
    mlpack::MakeAlias(inputTemp, data,
        FFT_FRAMES, FRAME_SIZE, 3, s * data.n_elem_slice + col * data.n_rows);
    for (size_t r = 0; r < FRAME_SIZE; r++) {
      frame[r] = inputTemp(0, r, 0);
    }
    spectrogram->pushFrame(frame);
  }
  spectrogram->updateWaterfall();
}

void InteractiveDebugger::hidePhoneMarkers(size_t start) {
  for (size_t i = start; i < clipPhoneMarkers.size(); i++)
    clipPhoneMarkers[i]->hide();
}

void InteractiveDebugger::placePhoneMarkers(const CPU_CUBE_TYPE& labels, size_t col) {
  size_t markCount = 0;
  size_t segmentStart = 0;
  size_t curPhone = labels(0, col, 0);
  for (size_t i = 1; i < labels.n_slices; i++) {
    size_t slicePhone = labels(0, col, i);
    if (slicePhone != curPhone) {
      // Get a label
      QLabel* label;
      if (markCount < clipPhoneMarkers.size()) {
        label = clipPhoneMarkers[markCount];
      } else {
        label = new QLabel(qtWindow);
        clipPhoneMarkers.push_back(label);
      }
      markCount++;
      // Set the text
      label->setAlignment(Qt::AlignCenter);
      label->resize(50, 20);
      label->setText(G_PS_C.xSampa(curPhone).c_str());
      label->show();
      // Place it at the midpoint of phone boundaries
      size_t midPt = (segmentStart + i) / 2;
      int midPos = spectrogram->getSliceX(midPt);
      label->move(midPos - 25, 70);

      segmentStart = i;
      curPhone = slicePhone;
    }
  }
  hidePhoneMarkers(markCount);
}

void InteractiveDebugger::findAndLoadClip() {
  static ClassifierHelper helper;
  static bool init = false;
  if (!init) {
    helper.initalize(sampleRate);
    init = true;
  }
  clipSlider->setValue(0);
  // Load clip
  std::vector<Phone> phones;
  currentClip = ds.findAndLoad(selectedPhoneme, sampleRate, phones);
  // Convert to frames
  size_t nFrames;
  std::vector<Frame> frames;
  Dataset::clipToFrames(currentClip, nFrames, frames, helper, phones);
  // Convert to cube
  CPU_CUBE_TYPE data, labels;
  Dataset::makeCubes(data, labels, 1, N_SLICES);
  size_t nSlices;
  ds.framesToCube(helper, frames, data, labels, 0, nFrames, nSlices);
  // Draw spectrogram
  loadCubeSpectrogram(data, 0);
  // Place markers
  placePhoneMarkers(labels, 0);
  // Provide audio to output
  {
    std::unique_lock lock(audioContainer->mtx);
    audioContainer->audio = currentClip.buffer;
    audioContainer->pointer = 0;
  }
}

void InteractiveDebugger::playSpeech(size_t phoneme) {
  std::unique_lock<std::mutex> lock(audioContainer->mtx);
  SpeechFrame sf;
  sf.phoneme = phoneme;
  // Write 0.8 seconds of audio from speech engine
  speechEngine->pushFrame(sf);
  audioContainer->audio.resize(sampleRate);
  speechEngine->writeBuffer(audioContainer->audio.data(), sampleRate * 0.8);
  // Write remaining 0.2 seconds of the end of the phoneme
  sf.phoneme = G_P_SIL_S;
  speechEngine->pushFrame(sf);
  speechEngine->writeBuffer(audioContainer->audio.data(), sampleRate * 0.2);

  audioContainer->pointer = 0;
}

void InteractiveDebugger::playSpeechSequence() {
  std::unique_lock<std::mutex> lock(audioContainer->mtx);
  SpeechFrame sf;
  // Allocate a minute of audio from speech engine
  audioContainer->audio.resize(sampleRate * 60);
  // Define sequence of phonemes to play
  std::vector<std::tuple<
      std::string /* Phoneme */,
      double /* Length (ms) */,
      double /* Pitch */ >> sequence = {
    //{"dey" , 0.600, 24},
    //{"ziy" , 0.600, 21},
    //{"dey" , 0.600, 17},
    //{"ziy" , 0.600, 12},
    //{"gih" , 0.200, 14},
    //{"vmiy", 0.200, 16},
    //{"yurr", 0.200, 17},
    //{"ah"  , 0.400, 14},
    //{"nsrr", 0.200, 17},
    //{"duw" , 1.200, 12}
    {"d", 0.05, 24},
    {"e", 0.15, 24},
    {"i", 0.40, 24},
    {"z", 0.20, 21},
    {"i", 0.40, 21},

    {"d", 0.05, 24},
    {"e", 0.15, 24},
    {"i", 0.40, 24},
    {"z", 0.20, 21},
    {"i", 0.40, 21},

    {"g", 0.05, 21},
    {"i", 0.15, 21},
    {"v", 0.05, 16},

    {"m", 0.05, 16},
    {"i", 0.10, 21},

    {"y", 0.05, 17},
    {"u", 0.05, 17},
    {"r", 0.10, 17},

    {"a", 0.40, 14},
    {"n", 0.10, 17},
    {"s", 0.05, 17},
    {"r", 0.05, 17},

    {"d", 0.10, 12},
    {"o", 1.10, 12},
    {".", 0.10, 20}
  };
  double time = 0;
  const double dt = 0.01; // Duration of one frame
  auto audioPtr = audioContainer->audio.data();
  size_t stepSamples = sampleRate * dt;

  for (std::tuple<std::string, double, double>& i : sequence) {
    std::string& phoneme = std::get<0>(i);
    double l = std::get<1>(i);
    double p = std::get<2>(i);
    double startT = time;
    
    // Translate current phoneme into speech signal
    size_t phoneId = G_PS_S.fromString(phoneme);
    while (time < startT + l) {
      sf.phoneme = phoneId;
      sf.pitch = p / 20;
      speechEngine->pushFrame(sf);
      speechEngine->writeBuffer(audioPtr, stepSamples);
      audioPtr += stepSamples;
      time += dt;
    }
  }
  audioContainer->pointer = 0;
}

void InteractiveDebugger::addSpeechEngine() {
  speechEngine = std::make_unique<SpeechEngineConcatenator>();
  speechEngine->setSampleRate(sampleRate)
      .setChannels(1)
      .setVolume(0.1)
      .configure("AERIS CV-VC ENG Kire 2.0/");
}
