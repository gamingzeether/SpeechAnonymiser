#define _DISABLE_VECTOR_ANNOTATION
#define _DISABLE_STRING_ANNOTATION

#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <rtaudio/RtAudio.h>
#include <pffft/pffft.h>
#include <thread>
#include "VulkanWindow.h"


typedef float INPUT_TYPE;
#define INPUT_FORMAT RTAUDIO_FLOAT32
#define INPUT_SCALE  1.0

typedef float OUTPUT_TYPE;
#define OUTPUT_FORMAT RTAUDIO_FLOAT32
#define OUTPUT_SCALE  1.0

#define INPUT_BUFFER_TIME 0.1
#define INPUT_BUFFER_SIZE 128
#define OUTPUT_BUFFER_SIZE 128

#define FFT_FRAME_SAMPLES 1024
#define FFT_VISUALIZATION_FRAMES 9

struct InputData {
    INPUT_TYPE* buffer;
    unsigned long bufferBytes;
    unsigned long totalFrames;
    unsigned int channels;
    unsigned int writeOffset;
};

struct OutputData {
    double* lastValues;
    unsigned int channels;
    unsigned long lastSample;
    InputData* input;
};

int processInput(void* /*outputBuffer*/, void* inputBuffer, unsigned int nBufferFrames,
    double /*streamTime*/, RtAudioStreamStatus /*status*/, void* data) {

    InputData* iData = (InputData*)data;

    for (int i = 0; i < nBufferFrames; i++) {
        unsigned int writePosition = (iData->writeOffset + i) % iData->totalFrames;
        iData->buffer[writePosition] = ((INPUT_TYPE*)inputBuffer)[i];
    }
    iData->writeOffset += nBufferFrames;

    return 0;
}

int processOutput(void* outputBuffer, void* /*inputBuffer*/, unsigned int nBufferFrames,
    double /*streamTime*/, RtAudioStreamStatus status, void* data) {

    OutputData* oData = (OutputData*)data;
    OUTPUT_TYPE* buffer = (OUTPUT_TYPE*)outputBuffer;

    if (status) {
        std::cout << "Stream underflow detected!\n";
    }

    InputData* iData = oData->input;
    unsigned int startSample = oData->lastSample;
    for (int i = 0; i < oData->channels; i++) {
        for (int j = 0; j < nBufferFrames; j++) {
            unsigned int readPosition = (startSample + j) % iData->totalFrames;
            INPUT_TYPE input = iData->buffer[readPosition];
            double value = (double)input / INPUT_SCALE;
            *buffer++ = (OUTPUT_TYPE)(value * OUTPUT_SCALE * 0.5);
        }
    }
    oData->lastSample = startSample + nBufferFrames;

    return 0;
}

void cleanupRtAudio(RtAudio audio) {
    if (audio.isStreamOpen()) {
        audio.closeStream();
    }
}

int main() {
    // Create and start input device
#pragma region Input
    RtAudio::StreamOptions inputFlags;
    inputFlags.flags |= RTAUDIO_NONINTERLEAVED;

    RtAudio inputAudio;

    RtAudio::StreamParameters inputParameters;
    unsigned int defaultInput = inputAudio.getDefaultInputDevice();
    if (defaultInput == 0) {
        std::cout << "No input devices available\n";
        return 0;
    }
    RtAudio::DeviceInfo inputInfo = inputAudio.getDeviceInfo(defaultInput);

    std::cout << "Using input: " << inputInfo.name << '\n';
    std::cout << "Sample rate: " << inputInfo.preferredSampleRate << '\n';

    inputParameters.deviceId = defaultInput;
    inputParameters.nChannels = 1; // inputInfo.inputChannels;
    unsigned int sampleRate = inputInfo.preferredSampleRate;
    unsigned int bufferFrames = INPUT_BUFFER_SIZE;

    InputData inputData;
    inputData.totalFrames = (unsigned long)(sampleRate * INPUT_BUFFER_TIME);
    inputData.channels = inputParameters.nChannels;
    unsigned long totalBytes;
    totalBytes = inputData.totalFrames * inputParameters.nChannels * sizeof(INPUT_TYPE);
    inputData.buffer = new INPUT_TYPE[totalBytes];
    for (int i = 0; i < inputData.totalFrames; i++) {
        inputData.buffer[i] = 0;
    }
    inputData.writeOffset = 0;

    if (inputAudio.openStream(NULL, &inputParameters, INPUT_FORMAT,
        sampleRate, &bufferFrames, &processInput, (void*)&inputData, &inputFlags)) {
        std::cout << inputAudio.getErrorText() << '\n';
        return 0; // problem with device settings
    }

    if (inputAudio.startStream()) {
        std::cout << inputAudio.getErrorText() << '\n';
        cleanupRtAudio(inputAudio);
        return 0;
    }
#pragma endregion


    // Create and start output device
#pragma region Output
    RtAudio::StreamOptions outputFlags;
    outputFlags.flags |= RTAUDIO_NONINTERLEAVED;
    RtAudio outputAudio;

    RtAudio::StreamParameters outputParameters;
    unsigned int defaultOutput = outputAudio.getDefaultOutputDevice();
    if (defaultOutput == 0) {
        std::cout << "No output devices available\n";
        return 0;
    }
    RtAudio::DeviceInfo outputInfo = outputAudio.getDeviceInfo(defaultOutput);

    std::cout << "Using output: " << outputInfo.name << '\n';
    std::cout << "Sample rate: " << outputInfo.preferredSampleRate << '\n';

    outputParameters.deviceId = defaultOutput;
    outputParameters.nChannels = outputInfo.outputChannels;
    unsigned int outputSampleRate = outputInfo.preferredSampleRate;
    unsigned int outputBufferFrames = OUTPUT_BUFFER_SIZE;

    OutputData outputData;
    outputData.lastValues = (double*)calloc(outputParameters.nChannels, sizeof(double));
    outputData.channels = outputParameters.nChannels;
    outputData.input = &inputData;


    if (outputAudio.openStream(&outputParameters, NULL, OUTPUT_FORMAT,
        outputSampleRate, &outputBufferFrames, &processOutput, (void*)&outputData, &outputFlags)) {
        std::cout << outputAudio.getErrorText() << '\n';
        return 0; // problem with device settings
    }

    if (outputAudio.startStream()) {
        std::cout << outputAudio.getErrorText() << '\n';
        cleanupRtAudio(outputAudio);
        return 0;
    }
#pragma endregion

    // Setup data visualization
    VulkanWindow app;
    app.initWindow();
    app.fftData.frames = FFT_VISUALIZATION_FRAMES;
    app.fftData.currentFrame = 0;
    app.fftData.frequencies = new float*[FFT_VISUALIZATION_FRAMES];
    for (int i = 0; i < FFT_VISUALIZATION_FRAMES; i++) {
        app.fftData.frequencies[i] = new float[FFT_FRAME_SAMPLES];
        for (int j = 0; j < FFT_FRAME_SAMPLES; j++) {
            app.fftData.frequencies[i][j] = 0.0;
        }
    }
    std::thread fft = std::thread([&app, &inputAudio, &inputData, &sampleRate] {
        // Setup FFT
        PFFFT_Setup* pffftSetup = pffft_new_setup(FFT_FRAME_SAMPLES, pffft_transform_t::PFFFT_REAL);
        float* fft_in = (float*)pffft_aligned_malloc(FFT_FRAME_SAMPLES * sizeof(float));
        float* fft_out = (float*)pffft_aligned_malloc(FFT_FRAME_SAMPLES * sizeof(float));
        float* fft_work = (float*)pffft_aligned_malloc(FFT_FRAME_SAMPLES * sizeof(float));
        float* window = new float[FFT_FRAME_SAMPLES];
        for (int i = 0; i < FFT_FRAME_SAMPLES; i++) {
            //window[i] = 1.0; // None
            window[i] = 0.5 * (1 - cos((6.28318530718 * i) / FFT_FRAME_SAMPLES)); // Hanning
        }

        while (!app.isOpen) {
            Sleep(10);
        }
        while (app.isOpen && inputAudio.isStreamRunning()) {
            unsigned int start = inputData.writeOffset - INPUT_BUFFER_SIZE;
            for (int i = 0; i < FFT_FRAME_SAMPLES; i++) {
                unsigned int readLocation = (start + i) % inputData.totalFrames;
                float data = inputData.buffer[readLocation];
                fft_in[i] = data * window[i];
            }

            pffft_transform_ordered(pffftSetup, fft_in, fft_out, fft_work, PFFFT_FORWARD);

            for (int i = 0; i < FFT_FRAME_SAMPLES; i++) {
                app.fftData.frequencies[app.fftData.currentFrame][i] = abs(fft_out[i]);
            }
            app.fftData.currentFrame = (app.fftData.currentFrame + 1) % FFT_VISUALIZATION_FRAMES;

            Sleep(20);
        }

        // Cleanup
        pffft_destroy_setup(pffftSetup);
        pffft_aligned_free(fft_in);
        pffft_aligned_free(fft_out);
        pffft_aligned_free(fft_work);
        });

    app.run();
    fft.join();

    return 0;
}
