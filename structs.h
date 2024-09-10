#pragma once

#include "define.h"

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
    //std::vector<float> imaginary;
    //std::vector<float> delta;
    float volume;
    size_t phone;

    void reset() {
        for (size_t i = 0; i < FRAME_SIZE; i++) {
            real[i] = 0;
            //imaginary[i] = 0;
            //delta[i] = 0;
        }
        volume = 0;
        phone = 0;
    }

    Frame() {
        real = std::vector<float>(FRAME_SIZE);
        //imaginary = std::vector<float>(FRAME_SIZE);
        //delta = std::vector<float>(FRAME_SIZE);
        volume = 0;
        phone = 0;
    }
};