#include "Spectrogram.hpp"

#include <math.h>

#include <QPainter>

#define POINT_WIDTH 2 // Width of each point in the spectrogram
#define POINT_HEIGHT 6 // Height of each point in the spectrogram
#define SPEC_WIDTH POINT_WIDTH * nFrames // Total width of spectrogram
#define SPEC_HEIGHT POINT_HEIGHT * FRAME_SIZE // Total height of spectrogram
#define COLORMAP_STEPS 256

Spectrogram::Spectrogram(QWidget* qtWindow, size_t width) : nFrames(width) {
  genColormap();

  waterfallPixmap = new QPixmap(SPEC_WIDTH, SPEC_HEIGHT);
  waterfallPixmap->fill(Qt::black);

  setParent(qtWindow);
  setPixmap(*waterfallPixmap);
}

void Spectrogram::pushFrame(std::array<float, FRAME_SIZE> frame) {
  auto lock = std::unique_lock<std::mutex>(waterfallMutex);
  for (float& f : frame)
    f = std::tanh(f / 50.0);
  waterfallBuffer.push_back(std::move(frame));
}

void Spectrogram::updateWaterfall() {
  auto lock = std::unique_lock<std::mutex>(waterfallMutex);
  size_t nFrames = waterfallBuffer.size();
  if (nFrames == 0)
    return;
  
  QPainter painter;
  painter.begin(waterfallPixmap);
  
  // Shift everything over by POINT_WIDTH * nFrames pixels
  int horizontalOffset = -POINT_WIDTH * nFrames;
  painter.drawTiledPixmap(horizontalOffset, 0, SPEC_WIDTH, SPEC_HEIGHT, *waterfallPixmap);
  for (size_t x = 0; x < nFrames; x++) {
    // Draw new frame
    std::array<float, FRAME_SIZE> currentFrame = waterfallBuffer[x];
    size_t realX = nFrames - nFrames + x;
    for (int y = 0; y < FRAME_SIZE; y++) {
      float val = currentFrame[FRAME_SIZE - 1 -y];
      double col = 0.5 * (1.0 + val); // [-1, 1] -> [0, 1]
      setSpectrogramPixel(painter, realX, y, col);
    }
  }
  waterfallBuffer.clear();
  lock.unlock();

  painter.end();

  setPixmap(*waterfallPixmap);
  update();
}

void Spectrogram::clearSpectrogram() {
  auto lock = std::unique_lock<std::mutex>(waterfallMutex);
  waterfallBuffer.clear();
  waterfallPixmap->fill(Qt::black);
  setPixmap(*waterfallPixmap);
  update();
}

int Spectrogram::getSliceX(size_t slice) {
  return slice * POINT_WIDTH + this->x();
}

void Spectrogram::setSpectrogramPixel(QPainter& painter, int x, int y, float val) {
  painter.fillRect(
      x * POINT_WIDTH,
      y * POINT_HEIGHT,
      POINT_WIDTH,
      POINT_HEIGHT,
      getColor(val));
}

void Spectrogram::genColormap() {
  auto lerp = [](double v0, double v1, double f) -> double {
    return (v1 - v0) * f + v0;
  };

  colormap = std::vector<QColor>(COLORMAP_STEPS);
  for (size_t i = 0; i < COLORMAP_STEPS; i++) {
    double rx = (double)i / (COLORMAP_STEPS - 1);
    // "cos" shape
    double c = (1.0 - std::cos(2.0 * M_PI * rx)) / 2.0;

    // Angle from -185 deg to 170 deg
    double h = lerp(-185, 170, rx);
    double h_rad = h * (M_PI / 180);

    // ab arc
    double a = c * std::cos(h_rad);
    double b = c * std::sin(h_rad);

    // Lightness from 2 to 99
    double j = lerp(2, 99, rx);

    // Convert JCh (aka LCh) to rgb
    // https://stackoverflow.com/a/75850608
    {
      double l = j;
      a *= 100;
      b *= 100;

      double xw = 0.94811;
      double yw = 1.00000;
      double zw = 1.07304;
      
      // Compute intermediate values
      double fy = (l + 16) / 116;
      double fx = fy + (a / 500);
      double fz = fy - (b / 200);
      
      // Compute XYZ values
      double x = xw * ((std::pow(fx, 3) > 0.008856) ? std::pow(fx, 3) : (((fx - 16) / 116) / 7.787));
      double y = yw * ((std::pow(fy, 3) > 0.008856) ? std::pow(fy, 3) : (((fy - 16) / 116) / 7.787));
      double z = zw * ((std::pow(fz, 3) > 0.008856) ? std::pow(fz, 3) : (((fz - 16) / 116) / 7.787));
      
      double r =  x * 3.2406 - y * 1.5372 - z * 0.4986;
      double g = -x * 0.9689 + y * 1.8758 + z * 0.0415;
      double b =  x * 0.0557 - y * 0.2040 + z * 1.0570;
      
      double rp = r > 0.0031308 ? 1.055 * std::pow(r, 1 / 2.4) - 0.055 : 12.92 * r;
      double gp = g > 0.0031308 ? 1.055 * std::pow(g, 1 / 2.4) - 0.055 : 12.92 * g;
      double bp = b > 0.0031308 ? 1.055 * std::pow(b, 1 / 2.4) - 0.055 : 12.92 * b;
      
      int i_r = std::max(std::min(rp, 1.0), 0.0) * 255;
      int i_g = std::max(std::min(gp, 1.0), 0.0) * 255;
      int i_b = std::max(std::min(bp, 1.0), 0.0) * 255;
      colormap[i] = QColor(i_r, i_g, i_b);
    }
  }
}

const QColor& Spectrogram::getColor(float value) {
  int index = value * COLORMAP_STEPS + 0.5;
  index = std::max(0, std::min(index, COLORMAP_STEPS - 1));
  return colormap[index];
}
