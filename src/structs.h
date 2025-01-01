#pragma once

#include "define.h"

#include <algorithm>
#include <vector>

struct InputData {
    INPUT_TYPE** buffer;
    size_t bufferBytes;
    size_t totalFrames;
    size_t channels;
    size_t writeOffset;
    size_t lastProcessed;

    InputData() {}
    InputData(size_t ch, size_t fr) {
        totalFrames = fr;
        channels = ch;
        buffer = new float* [ch];
        for (size_t i = 0; i < ch; i++) {
            buffer[i] = new float[fr];
            std::fill_n(buffer[i], fr, 0.0f);
        }
        writeOffset = 0;
    }
};

struct OutputData {
    double* lastValues;
    unsigned int channels;
    unsigned long lastSample;
    InputData* input;
    double scale;
};

struct Phone {
    size_t phonetic;
    double min;
    double max;
    unsigned int minIdx;
    unsigned int maxIdx;
};

struct Frame {
    std::vector<float> real;
    std::vector<float> avg; // Average of this frame and previous frames
    std::vector<float> delta;
    std::vector<float> accel;
    float volume;
    size_t phone;
    bool invalid;

    void reset() {
        for (size_t i = 0; i < FRAME_SIZE; i++) {
            real[i] = 0;
            avg[i] = 0;
            delta[i] = 0;
            accel[i] = 0;
        }
        volume = 0;
        phone = 0;
        invalid = false;
    }

    Frame() {
        real = std::vector<float>(FRAME_SIZE);
        avg = std::vector<float>(FRAME_SIZE);
        delta = std::vector<float>(FRAME_SIZE);
        accel = std::vector<float>(FRAME_SIZE);
        volume = 0;
        phone = 0;
        invalid = false;
    }
};

struct SpeechFrame {
    size_t phoneme;
};
