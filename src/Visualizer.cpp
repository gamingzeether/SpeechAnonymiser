#ifdef GUI
#include "Visualizer.hpp"

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <array>
#include <set>
#include <thread>
#include <math.h>

#include <QApplication>
#include <QLabel>
#include <QLineEdit>
#include <QDoubleValidator>
#include <QTimer>
#include <QPixmap>
#include <QPainter>
#include <QCheckBox>

#include "define.hpp"

#define SPEC_FRAMES 300 // Number of frames in spectrogram
#define SPEC_HEIGHT 6 // Height of each point in the spectrogram
#define COLORMAP_STEPS 256

void Visualizer::run() {
	genColormap();

    int argC = 0; // Dummy vars for QApplication
    char** argV;
    QApplication qtApp(argC, argV);
    QWidget qtWindow;
	initWindow(qtWindow);

	isOpen = true;
    qtApp.exec();
	cleanup();
	isOpen = false;
}

void Visualizer::updateSpectrogram() {
	if (!isOpen || !drawSpectrogram || spectrogramPixmap->isNull())
		return;
	QPainter painter;
	painter.begin(spectrogramPixmap);

	// Shift everything over by 1 pixel
	painter.drawTiledPixmap(-1, 0, SPEC_FRAMES, FRAME_SIZE * SPEC_HEIGHT, *spectrogramPixmap);
	// Draw new frame
	float* currentFrame = fftData.frequencies[fftData.currentFrame];
	for (int y = 0; y < FRAME_SIZE; y++) {
		float val = currentFrame[FRAME_SIZE - 1 -y];
		double col = 0.5 * (1.0 + std::tanh(val / 50.0));
		painter.setPen(getColor(col));
		for (int h = 0; h < SPEC_HEIGHT; h++) {
			painter.drawPoint(SPEC_FRAMES - 1, y * SPEC_HEIGHT + h);
		}
	}

	painter.end();
	spectrogramLabel->setPixmap(*spectrogramPixmap);
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
	spectrogramCheckbox->setText("Draw spectrogram");
	QObject::connect(spectrogramCheckbox, &QCheckBox::stateChanged, this, [&](int state){
		drawSpectrogram = (state == 2);
	});
    spectrogramCheckbox->show();

	spectrogramPixmap = new QPixmap(SPEC_FRAMES, FRAME_SIZE * SPEC_HEIGHT);
	spectrogramPixmap->fill(Qt::black);

	spectrogramLabel = new QLabel(&qtWindow);
    spectrogramLabel->move(10, 100);
	spectrogramLabel->setPixmap(*spectrogramPixmap);
    spectrogramLabel->show();

}

void Visualizer::cleanup() {

}

// Generate roseus colormap
// https://github.com/dofuuz/roseus/blob/main/roseus/generator.py
void Visualizer::genColormap() {
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

const QColor& Visualizer::getColor(float value) {
	return colormap[(size_t)(value * COLORMAP_STEPS + 0.5)];
}
#endif
