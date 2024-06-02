#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <rtaudio/RtAudio.h>
#include <rtaudio/rtaudio_c.h>

typedef float INPUT_TYPE;
#define INPUT_FORMAT RTAUDIO_FLOAT32
#define INPUT_SCALE  1.0

typedef float OUTPUT_TYPE;
#define OUTPUT_FORMAT RTAUDIO_FLOAT32
#define OUTPUT_SCALE  1.0

#define INPUT_BUFFER_TIME 5
#define INPUT_BUFFER_SIZE 2048
#define OUTPUT_BUFFER_SIZE 2048

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
    InputData* input;
};

int processInput(void* /*outputBuffer*/, void* inputBuffer, unsigned int nBufferFrames,
    double /*streamTime*/, RtAudioStreamStatus /*status*/, void* data) {

    InputData* iData = (InputData*)data;

    for (int i = 0; i < nBufferFrames; i++) {
        unsigned int writePosition = (iData->writeOffset + i) % iData->bufferBytes;
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
    for (int i = 0; i < oData->channels; i++) {
        for (int j = 0; j < nBufferFrames; j++) {
            unsigned int readPosition = (iData->writeOffset - nBufferFrames + j) % iData->bufferBytes;
            INPUT_TYPE input = iData->buffer[readPosition];
            double value = (double)input / INPUT_SCALE;
            *buffer++ = (OUTPUT_TYPE)(value * OUTPUT_SCALE * 0.5);
        }
    }

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

    inputParameters.deviceId = defaultInput;
    inputParameters.nChannels = inputInfo.inputChannels;
    unsigned int sampleRate = inputInfo.preferredSampleRate;
    unsigned int bufferFrames = INPUT_BUFFER_SIZE;

    InputData inputData;
    inputData.bufferBytes = bufferFrames * inputParameters.nChannels * sizeof(INPUT_TYPE);
    inputData.totalFrames = (unsigned long)(sampleRate * INPUT_BUFFER_TIME);
    inputData.channels = inputParameters.nChannels;
    unsigned long totalBytes;
    totalBytes = inputData.totalFrames * inputParameters.nChannels * sizeof(INPUT_TYPE);
    inputData.buffer = (INPUT_TYPE*)malloc(totalBytes);
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

    // Test input
    DWORD sleepTime = (DWORD)1000 * INPUT_BUFFER_TIME;
    unsigned int counter = 0;
    while (inputAudio.isStreamRunning()) {
        Sleep(sleepTime);
        std::cout << counter << "\n";
        counter++;
        //for (int i = 0; i < totalBytes; i += 100) {
        //    unsigned int readPos = (userData.writeOffset + i) % userData.bufferBytes;
        //    //std::cout << userData.buffer[readPos] << '\n';
        //}
    }

    cleanupRtAudio(inputAudio);
    cleanupRtAudio(outputAudio);

    return 0;
}
