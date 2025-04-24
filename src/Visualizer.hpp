#pragma once
#ifdef GUI
#include "define.hpp"

#include "common_inc.hpp"

// https://github.com/heitaoflower/vulkan-tutorial/blob/master/Tutorial24

#include <chrono>
#include <vector>
#include <array>
#include <string>
#include <mutex>

#include <QObject>
#include <QPixmap>
#include <QLabel>

#include "Utils/ClassifierHelper.hpp"

class Visualizer : QObject{
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
  void updateWaterfall();
  void setSpectrogramPixel(int x, int y, float val);
private:
  QLabel* peakAmountLabel;
  QLabel* waterfallLabel;
  QPixmap* waterfallPixmap;
  bool drawWaterfall = false;
  std::vector<QColor> colormap;
  std::vector<std::array<float, FRAME_SIZE>> waterfallBuffer;
  std::mutex waterfallMutex;

  void initWindow(QWidget& qtWindow);
  void cleanup();
  void genColormap();
  const QColor& getColor(float value);
};
#endif
