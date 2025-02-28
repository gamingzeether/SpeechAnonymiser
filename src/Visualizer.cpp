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

void Visualizer::run() {
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
		int col = 255 * 0.5 * (1.0 + std::tanh(val / 50.0));
		painter.setPen(QColor(col, col, col));
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
#endif
