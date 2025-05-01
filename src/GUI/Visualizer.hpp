#pragma once

#include "../common_inc.hpp"

#include <QObject>
#include <QPixmap>
#include <QLabel>

#include "Spectrogram.hpp"
#include "../Utils/ClassifierHelper.hpp"

class Visualizer : public QObject {
public:
  struct Frequencies {
    float** frequencies;
    int frames;
    int currentFrame;
  };

  bool isOpen = false;
  Frequencies fftData;

  ClassifierHelper* classifierHelper;

  void run();
  void updateWaterfall(std::array<float, FRAME_SIZE>& frame);
private:
  QLabel* peakAmountLabel;
  Spectrogram* waterfallLabel;
  bool drawWaterfall = false;

  void initWindow(QWidget& qtWindow);
  void cleanup();
};
