#include "define.h"

#include "common_inc.h"

#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <thread>
#include <filesystem>
#include <random>
#include <unordered_set>
#include <rtaudio/RtAudio.h>
#include <cargs.h>
#include "Visualizer.h"
#include "TSVReader.h"
#include "ModelSerializer.h"
#include "structs.h"
#include "PhonemeClassifier.h"

using namespace arma;
using namespace mlpack;

auto programStart = std::chrono::system_clock::now();
int SAMPLE_RATE = 16000;

PhonemeClassifier classifier;

static struct cag_option options[] = {
  {
    .identifier = 't',
    .access_letters = "t",
    .access_name = "train",
    .value_name = "PATH",
    .description = "Phoneme classifier training mode"},

  {
    .identifier = 'p',
    .access_letters = "p",
    .access_name = "preprocess",
    .value_name = "PATH",
    .description = "Preprocess training data"},

  {
    .identifier = 'h',
    .access_letters = "h",
    .access_name = "help",
    .description = "Shows the command help"}
};

template <typename T>
bool requestString(const std::string& request, std::string& out, T& value) {
    std::cout << request << std::endl << "Default: " << value << std::endl;
    std::getline(std::cin, out);
    std::cout << std::endl;
    return out != "";
}

void requestInput(const std::string& request, int& value) {
    std::string response;
    if (requestString(request, response, value)) {
        value = std::stoi(response);
    }
}

void requestInput(const std::string& request, float& value) {
    std::string response;
    if (requestString(request, response, value)) {
        value = std::stof(response);
    }
}

void requestInput(const std::string& request, double& value) {
    std::string response;
    if (requestString(request, response, value)) {
        value = std::stod(response);
    }
}

int processInput(void* /*outputBuffer*/, void* inputBuffer, unsigned int nBufferFrames,
    double /*streamTime*/, RtAudioStreamStatus /*status*/, void* data) {

    InputData* iData = (InputData*)data;

    if ((iData->lastProcessed - iData->writeOffset) % iData->totalFrames < nBufferFrames) {
        auto sinceStart = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::system_clock::now() - programStart);
        std::cout << sinceStart.count() << ": Stream overflow detected!\n";
        iData->writeOffset = (iData->lastProcessed + nBufferFrames + 128) % iData->totalFrames;
    }

    for (size_t i = 0; i < nBufferFrames; i++) {
        size_t writePosition = (iData->writeOffset + i) % iData->totalFrames;
        for (int j = 0; j < iData->channels; j++) {
            iData->buffer[j][writePosition] = ((INPUT_TYPE*)inputBuffer)[i * 2 + j];
        }
    }
    iData->writeOffset = (iData->writeOffset + nBufferFrames) % iData->totalFrames;

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
    size_t startSample = oData->lastSample;
    for (size_t i = 0; i < nBufferFrames; i++) {
        size_t readPosition = (startSample + i) % iData->totalFrames;
        for (size_t j = 0; j < oData->channels; j++) {
            INPUT_TYPE input = iData->buffer[j][readPosition];
            *buffer++ = (OUTPUT_TYPE)((input * OUTPUT_SCALE) / INPUT_SCALE);
        }
    }
    oData->lastSample = (startSample + nBufferFrames) % iData->totalFrames;
    iData->lastProcessed = oData->lastSample;

    return 0;
}

void cleanupRtAudio(RtAudio audio) {
    if (audio.isStreamOpen()) {
        audio.closeStream();
    }
}

void startFFT(InputData& inputData) {
    float activationThreshold = 0.01;
    if (classifier.ready) {
        std::cout << "Model successfully loaded" << std::endl;
        requestInput("Set activation threshold", activationThreshold);
    } else {
        std::cout << "Model could not be loaded, disabling classification" << std::endl;
        activationThreshold = std::numeric_limits<float>::max();
    }

    Visualizer app;
    app.initWindow();
    std::cout << "Starting visualizer" << std::endl;
    app.fftData.frames = FFT_FRAMES;
    app.fftData.currentFrame = 0;
    app.fftData.frequencies = new float* [FFT_FRAMES];
    for (size_t i = 0; i < FFT_FRAMES; i++) {
        app.fftData.frequencies[i] = new float[FFT_REAL_SAMPLES];
        for (size_t j = 0; j < FFT_REAL_SAMPLES; j++) {
            app.fftData.frequencies[i][j] = 0.0;
        }
    }
    std::thread fft = std::thread([&app, &inputData, &activationThreshold] {
        // Setup classifier
        mat data(classifier.getInputSize(), 1);
        mat out(1, 1);

        // Wait for visualization to open
        while (!app.isOpen) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::cout << "Starting FFT thread processing" << std::endl;
        std::vector<Frame> frames = std::vector<Frame>(FFT_FRAMES);
        size_t currentFrame = 0;
        size_t lastSampleStart = 0;
        while (app.isOpen) {
            Frame& frame = frames[currentFrame];
            Frame& prevFrame = frames[(currentFrame + FFT_FRAMES - 1) % FFT_FRAMES];
            // Wait for enough samples to be recorded to pass to FFT
            while ((inputData.writeOffset - lastSampleStart) % inputData.totalFrames < FFT_FRAME_SAMPLES) {
                //auto start = std::chrono::high_resolution_clock::now();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                //auto actual = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
                //std::cout << actual.count() << '\n';
            }
            lastSampleStart = (lastSampleStart + FFT_FRAME_SPACING) % inputData.totalFrames;
            // Do FFT stuff
            classifier.processFrame(frame, inputData.buffer[0], lastSampleStart, inputData.totalFrames, prevFrame);

            // Write FFT output to visualizer
            app.fftData.currentFrame = (app.fftData.currentFrame + 1) % FFT_FRAMES;
            memcpy(app.fftData.frequencies[app.fftData.currentFrame], frame.real.data(), sizeof(float) * FFT_REAL_SAMPLES);

            // Pass data to neural network
            if (frame.volume > activationThreshold) {
                classifier.writeInput(frames, currentFrame, data, 0);
                size_t phoneme = classifier.classify(data);
                std::cout << classifier.getPhonemeString(phoneme) << std::endl;
            }
            currentFrame = (currentFrame + 1) % FFT_FRAMES;
        }
        });

#ifdef _CONSOLE
#ifdef HIDE_CONSOLE
    ShowWindow(GetConsoleWindow(), SW_HIDE);
#else
    ShowWindow(GetConsoleWindow(), SW_SHOW);
#endif
#endif

    app.run();
    fft.join();
}

int commandHelp() {
    printf("Usage: SpeechAnonymiser [OPTION]...\n\n");
    cag_option_print(options, CAG_ARRAY_SIZE(options), stdout);
    return 0;
}

int commandTrain(const std::string& path) {
    std::cout << "Training mode\n";

    int batchSize = 100;
    requestInput("Set batch size", batchSize);
    if (batchSize <= 1) {
        throw("Out of range");
    }

    int epochs = 5;
    requestInput("Set number of epochs", epochs);
    if (epochs <= 0) {
        throw("Out of range");
    }

    double stepSize = STEP_SIZE;
    requestInput("Set training rate", stepSize);
    if (stepSize <= 0) {
        throw("Out of range");
    }

    classifier.train(path, batchSize, epochs, stepSize);

    return 0;
}

int commandPreprocess(const std::string& path) {
    classifier.preprocessDataset(path);

    return 0;
}

int commandDefault() {
#pragma region Input and output selector
    RtAudio audioQuery;
    auto devices = audioQuery.getDeviceIds();
    unsigned int inDevice = audioQuery.getDefaultInputDevice();
    if (inDevice == 0) {
        std::cout << "No input devices available\n";
        return 0;
    }
    unsigned int outDevice = audioQuery.getDefaultOutputDevice();
    if (outDevice == 0) {
        std::cout << "No output devices available\n";
        return 0;
    }
    std::vector<RtAudio::DeviceInfo> inputDevices = std::vector<RtAudio::DeviceInfo>();
    std::vector<RtAudio::DeviceInfo> outputDevices = std::vector<RtAudio::DeviceInfo>();
    int inIdx = 0;
    int outIdx = 0;
    for (size_t i = 0; i < devices.size(); i++) {
        unsigned int deviceId = devices[i];
        RtAudio::DeviceInfo deviceInfo = audioQuery.getDeviceInfo(deviceId);
        if (deviceInfo.inputChannels > 0) {
            if (deviceInfo.isDefaultInput) {
                inIdx = inputDevices.size();
            }
            inputDevices.push_back(deviceInfo);
        }
        if (deviceInfo.outputChannels > 0) {
            if (deviceInfo.isDefaultOutput) {
                outIdx = outputDevices.size();
            }
            outputDevices.push_back(deviceInfo);
        }
    }

    std::string response;
    for (size_t i = 0; i < inputDevices.size(); i++) {
        std::cout << i << ": " << inputDevices[i].name << std::endl;
    }
    requestInput("Select input device", inIdx);
    if (inIdx < 0 || inIdx >= inputDevices.size()) {
        throw("Out of range");
    } else {
        inDevice = inputDevices[inIdx].ID;
    }

    for (size_t i = 0; i < outputDevices.size(); i++) {
        std::cout << i << ": " << outputDevices[i].name << std::endl;
    }
    requestInput("Select output device", outIdx);
    if (outIdx < 0 || outIdx >= outputDevices.size()) {
        throw("Out of range");
    } else {
        outDevice = outputDevices[outIdx].ID;
    }
#pragma endregion

    // Create and start input device
#pragma region Input
    RtAudio::StreamOptions inputFlags;
    inputFlags.flags |= RTAUDIO_NONINTERLEAVED;

    RtAudio inputAudio;

    RtAudio::StreamParameters inputParameters;

    RtAudio::DeviceInfo inputInfo = inputAudio.getDeviceInfo(inDevice);

    inputParameters.deviceId = inDevice;
    inputParameters.nChannels = inputInfo.inputChannels;
    unsigned int sampleRate = SAMPLE_RATE;
    unsigned int bufferFrames = INPUT_BUFFER_SIZE;

    std::cout << "Using input: " << inputInfo.name << '\n';
    std::cout << "Sample rate: " << sampleRate << '\n';
    std::cout << "Channels: " << inputInfo.inputChannels << '\n';

    InputData inputData = InputData(inputParameters.nChannels, sampleRate * INPUT_BUFFER_TIME);

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
    RtAudio::DeviceInfo outputInfo = outputAudio.getDeviceInfo(outDevice);

    outputParameters.deviceId = outDevice;
    outputParameters.nChannels = outputInfo.outputChannels;
    unsigned int outputSampleRate = SAMPLE_RATE;
    unsigned int outputBufferFrames = OUTPUT_BUFFER_SIZE;

    std::cout << "Using output: " << outputInfo.name << '\n';
    std::cout << "Sample rate: " << outputSampleRate << '\n';
    std::cout << "Channels: " << outputInfo.outputChannels << '\n';

    OutputData outputData = OutputData();
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

    std::cout << std::endl;

    // Setup data visualization
    startFFT(inputData);

    cleanupRtAudio(inputAudio);
    cleanupRtAudio(outputAudio);

    return 0;
}

int main(int argc, char** argv) {
    srand(static_cast <unsigned> (time(0)));

    requestInput("Select sample rate", SAMPLE_RATE);
    classifier.initalize(SAMPLE_RATE, true);

    float gain = 1;
    requestInput("Set gain", gain);
    classifier.setGain(gain);

    cag_option_context context;
    cag_option_init(&context, options, CAG_ARRAY_SIZE(options), argc, argv);
    int error = 0;
    bool doDefault = true;
    while (cag_option_fetch(&context)) {
        doDefault = false;
        switch (cag_option_get_identifier(&context)) {
        case 't':
            error = commandTrain(cag_option_get_value(&context));
            break;
        case 'p':
            error = commandPreprocess(cag_option_get_value(&context));
            break;
        case 'h':
            error = commandHelp();
            break;
        }
    }
    if (doDefault) {
        error = commandDefault();
    }

    classifier.destroy();

    return error;
}
