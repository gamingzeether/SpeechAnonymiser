#pragma once

#include "../common_inc.hpp"

#include <mutex>

#include <QLabel>
#include <QPixmap>

#include "../define.hpp"

class Spectrogram : public QLabel {
public:
  Spectrogram(QWidget* qtWindow, size_t width);
  void pushFrame(std::array<float, FRAME_SIZE> frame);
  void updateWaterfall();
  void clearSpectrogram();
  size_t getWidth() { return nFrames; };
  int getSliceX(size_t slice);
private:
  Spectrogram();
  void setSpectrogramPixel(QPainter& painter, int x, int y, float val);
  // Generate roseus colormap
  // https://github.com/dofuuz/roseus/blob/main/roseus/generator.py
  void genColormap();
  const QColor& getColor(float value);

  size_t nFrames;
  QPixmap* waterfallPixmap;
  std::vector<QColor> colormap;
  std::vector<std::array<float, FRAME_SIZE>> waterfallBuffer;
  std::mutex waterfallMutex;
};
