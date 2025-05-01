#include "Visualizer.hpp"

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <array>
#include <set>
#include <thread>

#include <QApplication>
#include <QLabel>
#include <QLineEdit>
#include <QDoubleValidator>
#include <QTimer>
#include <QCheckBox>

#include "../define.hpp"

void Visualizer::run() {
  // Dummy vars for QApplication
  int argC = 0;
  char** argV;
  QApplication qtApp(argC, argV);
  QWidget qtWindow;
  initWindow(qtWindow);

  isOpen = true;
  qtApp.exec();
  cleanup();
  isOpen = false;
}

void Visualizer::updateWaterfall(std::array<float, FRAME_SIZE>& frame) {
  // We're just writing to a buffer here
  // Actual drawing is done in waterfallTimer
  if (!isOpen || !drawWaterfall)
    return;
  
  waterfallLabel->pushFrame(std::move(frame));
}

void Visualizer::initWindow(QWidget& qtWindow) {
  qtWindow.resize(320, 240);
  qtWindow.show();
  qtWindow.setWindowTitle("Visualizer");

  // ==================== Gain ====================
  QLabel* gainLabel = new QLabel("Gain", &qtWindow);
  gainLabel->move(10, 10);
  gainLabel->show();

  QLineEdit* gainTextEdit = new QLineEdit("1.0", &qtWindow);
  gainTextEdit->setValidator(new QDoubleValidator(0.01, 1000, 1, &qtWindow));
  QObject::connect(gainTextEdit, &QLineEdit::textChanged, this, [&](const QString &text){
    float gain = text.toFloat();
    (*classifierHelper).setGain(gain);
  });
  gainTextEdit->move(50, 12);
  gainTextEdit->show();

  // ==================== Peak ====================
  QLabel* peakLabel = new QLabel("Peak", &qtWindow);
  peakLabel->move(10, 40);
  peakLabel->show();

  peakAmountLabel = new QLabel("-0.1", &qtWindow);
  peakAmountLabel->setMinimumWidth(150);
  peakAmountLabel->move(50, 40);
  peakAmountLabel->show();

  QTimer* peakTimer = new QTimer(&qtWindow);
  QObject::connect(peakTimer, &QTimer::timeout, this, [&](){
    float peak = -9e99;
    for (int i = 0; i < fftData.frames; i++) {
      peak = std::max(peak, fftData.frequencies[i][0]);
    }
    peakAmountLabel->setText(std::to_string(peak).c_str());
  });
  peakTimer->start(50);

  // ==================== Spectrogram ====================
  QCheckBox* spectrogramCheckbox = new QCheckBox(&qtWindow);
  spectrogramCheckbox->move(10, 70);
  spectrogramCheckbox->setText("Draw waterfall plot");
  QObject::connect(spectrogramCheckbox, &QCheckBox::stateChanged, this, [&](int state){
    drawWaterfall = (state == 2);
    waterfallLabel->clearSpectrogram();
  });
  spectrogramCheckbox->show();

  waterfallLabel = new Spectrogram(&qtWindow, 150);
  waterfallLabel->move(10, 100);
  waterfallLabel->show();

  // Timer to update waterfall
  QTimer* waterfallTimer = new QTimer(&qtWindow);
  QObject::connect(waterfallTimer, &QTimer::timeout, this, [&](){
    if (!drawWaterfall)
      return;
    waterfallLabel->updateWaterfall();
  });
  int targetFrameRate = 50;
  waterfallTimer->start(1000 / targetFrameRate);
}

void Visualizer::cleanup() {

}
