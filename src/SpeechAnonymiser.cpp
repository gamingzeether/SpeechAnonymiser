#include "define.h"

#include "common_inc.h"

#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <thread>
#include <filesystem>
#include <random>
#include <unordered_set>
#include <format>
#include <optional>
#include <rtaudio/RtAudio.h>
#include <cargs.h>
#include "Visualizer.h"
#include "TSVReader.h"
#include "ModelSerializer.h"
#include "structs.h"
#include "PhonemeClassifier.h"
#include "ClassifierHelper.h"
#include "Dataset.h"
#include "Logger.h"
#include "SpeechEngineConcatenator.h"

const bool outputPassthrough = true;

auto programStart = std::chrono::system_clock::now();
int sampleRate = 16000;

PhonemeClassifier classifier;

std::unique_ptr<SpeechEngine> speechEngine;

Logger logger;

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
    .description = "Shows the command help"},

  {
    .identifier = 'w',
    .access_letters = "w",
    .value_name = "PATH",
    .description = "Work directory"},

  {
    .identifier = 'd',
    .access_letters = "d",
    .value_name = "PATH",
    .description = "MFA dictionary path"},

  {
    .identifier = 'a',
    .access_letters = "a",
    .value_name = "PATH",
    .description = "MFA acoustic model path"},

  {
    .identifier = 'o',
    .access_letters = "o",
    .value_name = "PATH",
    .description = "Output directory"},

  {
    .identifier = '\\',
    .access_name = "interactive",
    .value_name = "PATH",
    .description = "Used for development"
}
};

struct AudioContainer {
    std::vector<float> audio;
    size_t pointer;
    std::mutex mtx;
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

    if (outputPassthrough) {
        if ((iData->lastProcessed - iData->writeOffset) % iData->totalFrames < nBufferFrames) {
            auto sinceStart = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::system_clock::now() - programStart);
            std::cout << sinceStart.count() << ": Stream overflow detected!\n";
            iData->writeOffset = (iData->lastProcessed + nBufferFrames + 128) % iData->totalFrames;
        }
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

    if (!outputPassthrough) {
        speechEngine->writeBuffer((OUTPUT_TYPE*)outputBuffer, nBufferFrames);
    } else {
        InputData* iData = oData->input;
        size_t startSample = oData->lastSample;
        double scale = oData->scale;
        int totalFrames = iData->totalFrames * scale;
        for (size_t i = 0; i < nBufferFrames; i++) {
            size_t readPosition = (startSample + i) % totalFrames;
            for (size_t j = 0; j < oData->channels; j++) {
                INPUT_TYPE input;
                double realRead = (double)readPosition / scale;
                double floor = std::floor(realRead);
                OUTPUT_TYPE p1 = iData->buffer[j][(int)floor];
                OUTPUT_TYPE p2 = iData->buffer[j][(int)(std::ceil(realRead) + 0.1)];
                input = std::lerp(p1, p2, realRead - floor);
                *buffer++ = (OUTPUT_TYPE)((input * OUTPUT_SCALE) / INPUT_SCALE);
            }
        }
        oData->lastSample = (startSample + nBufferFrames) % totalFrames;
        iData->lastProcessed = (int)(oData->lastSample / scale);
    }

    return 0;
}

int oneshotOutput(void* outputBuffer, void* /*inputBuffer*/, unsigned int nBufferFrames,
    double /*streamTime*/, RtAudioStreamStatus status, void* data) {

    OUTPUT_TYPE* buffer = (OUTPUT_TYPE*)outputBuffer;

    AudioContainer& container = *(AudioContainer*)data;
    std::unique_lock<std::mutex> lock(container.mtx);
    size_t size = container.audio.size();
    size_t& ptr = container.pointer;
    for (size_t i = 0; i < nBufferFrames; i++) {
        float data = (ptr + i < size) ? container.audio[ptr++] : 0;
        for (size_t j = 0; j < 2; j++) {
            *buffer++ = data;
        }
    }
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
        logger.log("Model successfully loaded", Logger::INFO);
        requestInput("Set activation threshold", activationThreshold);
    } else {
        logger.log("Model could not be loaded, disabling classification", Logger::WARNING);
        //activationThreshold = std::numeric_limits<float>::max();
    }

    Visualizer app;
    app.initWindow();
    logger.log("Starting visualizer", Logger::INFO);
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
        CPU_MAT_TYPE data(classifier.getInputSize(), 1);
        CPU_MAT_TYPE out(1, 1);

        // Wait for visualization to open
        while (!app.isOpen) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        logger.log("Starting FFT thread processing", Logger::VERBOSE);
        std::vector<Frame> frames = std::vector<Frame>(FFT_FRAMES);
        for (Frame& f : frames) {
            f.reset();
        }
        size_t currentFrame = 0;
        size_t lastSampleStart = 0;

        //std::printf("Maximum time per frame: %fms\n", (1000.0 * FFT_FRAME_SPACING) / classifier.getSampleRate());

        ClassifierHelper& helper = ClassifierHelper::instance();
        SpeechFrame speechFrame;
        const size_t silencePhoneme = helper.phonemeSet[helper.customHasher(L"æ")];
        speechFrame.phoneme = silencePhoneme;
        int count = 0;
        // Wait for enough samples to be recorded to pass to FFT
        while ((inputData.writeOffset - lastSampleStart) % inputData.totalFrames < FFT_REAL_SAMPLES) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        while (app.isOpen) {
            Frame& frame = frames[currentFrame];
            // Wait for FFT_FRAME_SPACING new samples
            //auto start = std::chrono::high_resolution_clock::now();
            while ((inputData.writeOffset - lastSampleStart) % inputData.totalFrames < FFT_FRAME_SPACING) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            //auto actual = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
            //std::cout << actual.count() << '\n';
            lastSampleStart = (lastSampleStart + FFT_FRAME_SPACING) % inputData.totalFrames;
            // Do FFT stuff
            helper.processFrame(inputData.buffer[0], lastSampleStart, inputData.totalFrames, frames, currentFrame);

            // Write FFT output to visualizer
            app.fftData.currentFrame = currentFrame;
            memcpy(app.fftData.frequencies[currentFrame], frame.avg.data(), sizeof(float) * FFT_REAL_SAMPLES);

            // Pass data to neural network
            bool activity = false;
            for (int i = 0; i < ACTIVITY_WIDTH; i++) {
                Frame& activityFrame = frames[(currentFrame + FFT_FRAMES - i) % FFT_FRAMES];
                if (activityFrame.volume >= activationThreshold) {
                    activity = true;
                    break;
                }
            }

            // Classify
            if (activity) {
                count++;
                if (count > INFERENCE_FRAMES) {
                    count = 0;
                    helper.writeInput<CPU_MAT_TYPE>(frames, currentFrame, data, 0);

                    //auto classifyStart = std::chrono::high_resolution_clock::now();
                    size_t phoneme = classifier.classify(data);
                    speechFrame.phoneme = phoneme;
                    //auto classifyDuration = std::chrono::high_resolution_clock::now() - classifyStart;

                    std::cout << classifier.getPhonemeString(phoneme) << std::endl;
                    //std::cout << std::chrono::duration<double>(classifyDuration).count() * 1000 << " ms\n";
                }
            } else {
                speechFrame.phoneme = silencePhoneme;
            }
            speechEngine->pushFrame(speechFrame);
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
    logger.log("Training mode", Logger::INFO);

    int examples = 10000;
    requestInput("Set examples per phoneme", examples);
    logger.log(std::format("Set examples count: {}", examples), Logger::VERBOSE);
    if (examples <= 0) {
        throw("Out of range");
    }

    int epochs = 100;
    requestInput("Set number of epochs", epochs);
    logger.log(std::format("Set epochs: {}", epochs), Logger::VERBOSE);
    if (epochs <= 0) {
        throw("Out of range");
    }

    //double stepSize = STEP_SIZE;
    //requestInput("Set training rate", stepSize);
    //logger.log(std::format("Set training rate: {}", stepSize), Logger::VERBOSE);
    //if (stepSize <= 0) {
    //    throw("Out of range");
    //}

    classifier.train(path, examples, epochs);

    return 0;
}

int commandPreprocess(const std::string& path, const std::string& workDir, const std::string& dictPath, const std::string& acousticPath, const std::string& outputDir) {
    Dataset ds = Dataset();
    ds.preprocessDataset(path, workDir, dictPath, acousticPath, outputDir);

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

    RtAudio inputAudio;

    RtAudio::StreamParameters inputParameters;

    RtAudio::DeviceInfo inputInfo = inputAudio.getDeviceInfo(inDevice);

    inputParameters.deviceId = inDevice;
    inputParameters.nChannels = inputInfo.inputChannels;
    unsigned int inputSampleRate = sampleRate;
    unsigned int bufferFrames = INPUT_BUFFER_SIZE;

    logger.log(std::format("Using input: {}", inputInfo.name), Logger::INFO);
    logger.log(std::format("Sample rate: {}", inputSampleRate), Logger::INFO);
    logger.log(std::format("Channels: {}", inputInfo.inputChannels), Logger::INFO);

    InputData inputData = InputData(inputParameters.nChannels, inputSampleRate * INPUT_BUFFER_TIME);

    if (inputAudio.openStream(NULL, &inputParameters, INPUT_FORMAT,
        inputSampleRate, &bufferFrames, &processInput, (void*)&inputData, &inputFlags)) {
        std::cout << inputAudio.getErrorText() << '\n';
        return 0; // problem with device settings
    }
#pragma endregion


    // Create and start output device
#pragma region Output
    RtAudio::StreamOptions outputFlags;
    RtAudio outputAudio;

    RtAudio::StreamParameters outputParameters;
    RtAudio::DeviceInfo outputInfo = outputAudio.getDeviceInfo(outDevice);

    outputParameters.deviceId = outDevice;
    outputParameters.nChannels = outputInfo.outputChannels;
    unsigned int outputSampleRate = 44100;
    unsigned int outputBufferFrames = OUTPUT_BUFFER_SIZE;


    logger.log(std::format("Using output: {}", outputInfo.name), Logger::INFO);
    logger.log(std::format("Sample rate: {}", outputSampleRate), Logger::INFO);
    logger.log(std::format("Channels: {}", outputInfo.outputChannels), Logger::INFO);

    OutputData outputData = OutputData();
    outputData.lastValues = (double*)calloc(outputParameters.nChannels, sizeof(double));
    outputData.channels = outputParameters.nChannels;
    outputData.input = &inputData;
    outputData.scale = (double)outputSampleRate / inputSampleRate;

    if (outputAudio.openStream(&outputParameters, NULL, OUTPUT_FORMAT,
        outputSampleRate, &outputBufferFrames, &processOutput, (void*)&outputData, &outputFlags)) {
        std::cout << outputAudio.getErrorText() << '\n';
        return 0; // problem with device settings
    }
#pragma endregion

    // Initalize speech engine
    auto tmp = SpeechEngineConcatenator();
    tmp.setSampleRate(outputSampleRate)
        .setChannels(2)
        .setVolume(0.05)
        .configure("AERIS CV-VC ENG Kire 2.0/");
    speechEngine = std::unique_ptr<SpeechEngine>(&tmp);

    if (inputAudio.startStream()) {
        std::cout << inputAudio.getErrorText() << '\n';
        cleanupRtAudio(inputAudio);
        return 0;
    }

    if (outputAudio.startStream()) {
        std::cout << outputAudio.getErrorText() << '\n';
        cleanupRtAudio(outputAudio);
        return 0;
    }

    std::cout << std::endl;

    // Setup data visualization
    startFFT(inputData);

    return 0;
}

// For development - listening to and comparing sound clips/phonemes
int commandInteractive(const std::string& path) {
    AudioContainer ac;
    ac.audio = std::vector<float>(1);
    ac.pointer = 0;

#pragma region Output
    RtAudio audioQuery;
    auto devices = audioQuery.getDeviceIds();
    unsigned int outDevice = audioQuery.getDefaultOutputDevice();
    if (outDevice == 0) {
        std::cout << "No output devices available\n";
        return 0;
    }
    std::vector<RtAudio::DeviceInfo> outputDevices = std::vector<RtAudio::DeviceInfo>();
    int inIdx = 0;
    int outIdx = 0;
    for (size_t i = 0; i < devices.size(); i++) {
        unsigned int deviceId = devices[i];
        RtAudio::DeviceInfo deviceInfo = audioQuery.getDeviceInfo(deviceId);
        if (deviceInfo.outputChannels > 0) {
            if (deviceInfo.isDefaultOutput) {
                outIdx = outputDevices.size();
            }
            outputDevices.push_back(deviceInfo);
        }
    }

    std::string response;

    for (size_t i = 0; i < outputDevices.size(); i++) {
        std::cout << i << ": " << outputDevices[i].name << std::endl;
    }
    requestInput("Select output device", outIdx);
    if (outIdx < 0 || outIdx >= outputDevices.size()) {
        throw("Out of range");
    } else {
        outDevice = outputDevices[outIdx].ID;
    }

    RtAudio::StreamOptions outputFlags;
    RtAudio outputAudio;

    RtAudio::StreamParameters outputParameters;
    RtAudio::DeviceInfo outputInfo = outputAudio.getDeviceInfo(outDevice);

    outputParameters.deviceId = outDevice;
    outputParameters.nChannels = outputInfo.outputChannels;
    unsigned int outputSampleRate = 44100;
    unsigned int outputBufferFrames = OUTPUT_BUFFER_SIZE;


    logger.log(std::format("Using output: {}", outputInfo.name), Logger::INFO);
    logger.log(std::format("Sample rate: {}", outputSampleRate), Logger::INFO);
    logger.log(std::format("Channels: {}", outputInfo.outputChannels), Logger::INFO);

    OutputData outputData = OutputData();
    outputData.lastValues = (double*)calloc(outputParameters.nChannels, sizeof(double));
    outputData.channels = outputParameters.nChannels;
    outputData.input = NULL;
    outputData.scale = (double)outputSampleRate / outputSampleRate;

    if (outputAudio.openStream(&outputParameters, NULL, OUTPUT_FORMAT,
        outputSampleRate, &outputBufferFrames, &oneshotOutput, (void*)&ac, &outputFlags)) {
        std::cout << outputAudio.getErrorText() << '\n';
        return 0; // problem with device settings
    }
#pragma endregion

    auto tmp = SpeechEngineConcatenator();
    tmp.setSampleRate(outputSampleRate)
        .setChannels(1)
        .setVolume(0.1)
        .configure("AERIS CV-VC ENG Kire 2.0/");
    speechEngine = std::unique_ptr<SpeechEngine>(&tmp);

    Dataset ds = Dataset(path + "/test.tsv", outputSampleRate, path);

    if (outputAudio.startStream()) {
        std::cout << outputAudio.getErrorText() << '\n';
        cleanupRtAudio(outputAudio);
        return -1;
    }

    std::string command;
    std::string prefix = "~";
    std::vector<float> clipAudio;
    std::vector<Phone> clipPhones;
    while (true) {
        std::printf("%s: ", prefix);
        std::getline(std::cin, command);
        if (command == "voicebank" && prefix == "~") {
            prefix = "voicebank";
        } else if (command == "dataset" && prefix == "~") {
            prefix = "dataset";
        } else if (command == "exit") {
            prefix = "~";
        } else if (command != "" && prefix == "clip") {
            int index = std::stoi(command);
            if (0 <= index && index < clipPhones.size()) {
                const Phone& p = clipPhones[index];
                int buffer = 0.2 * sampleRate;
                int start = std::max(0, (int)p.minIdx - buffer);
                int end = std::min((int)(clipAudio.size() - 1), (int)p.maxIdx + buffer);
                int len = end - start;
                std::vector<float> aud = std::vector<float>(len);
                for (int i = 0; i < len; i++) {
                    float volume = (i + 4000 < len) ? 1 : (len - i) / 4000.0f;
                    aud[i] = clipAudio[start + i] * volume;
                }
                std::unique_lock<std::mutex> lock(ac.mtx);
                ac.audio = std::move(aud);
                ac.pointer = 0;
            } else if (index == -1) {
                std::unique_lock<std::mutex> lock(ac.mtx);
                ac.audio = clipAudio;
                ac.pointer = 0;
            }
        } else if (command != "" && prefix == "voicebank") {
            std::unique_lock<std::mutex> lock(ac.mtx);
            SpeechFrame sf;
            sf.phoneme = std::stoi(command);
            speechEngine->pushFrame(sf);
            ac.audio.resize(outputSampleRate / 2);
            speechEngine->writeBuffer(ac.audio.data(), outputSampleRate / 2);
            ac.pointer = 0;
        } else if (command != "" && prefix == "dataset") {
            size_t targetPhoneme = std::stoull(command);
            if (targetPhoneme < ClassifierHelper::instance().inversePhonemeSet.size()) {
                TSVReader::TSVLine tsv;
                clipAudio = ds._findAndLoad(path, targetPhoneme, outputSampleRate, tsv, clipPhones);
                prefix = "clip";
                std::printf("%s\n", tsv.PATH.c_str());
                for (int i = 0; i < clipPhones.size(); i++) {
                    const Phone& p = clipPhones[i];
                    std::printf("%d  %s: %.2f, %.2f\n", i, ClassifierHelper::instance().inversePhonemeSet[p.phonetic].c_str(), p.min, p.max);
                }
            } else {
                std::printf("Out of range\n");
            }
        } else {
            std::printf("Invalid command\n");
        }
    }
    return 0;
}

bool tryMakeDir(std::string path, bool fatal = true) {
    if (!std::filesystem::create_directories(path)) {
        if (std::filesystem::exists(path)) {
            return true;
        }
        if (fatal) {
            throw("Failed to make directory");
        }
        return false;
    }
    return true;
}

void initClassifier(int argc, char* argv[]) {
    std::string launchString = "";
    for (int i = 0; i < argc; i++) {
        if (i > 0) {
            launchString += " ";
        }
        launchString += argv[i];
    }
    logger.log(std::format("Launch args: {}", launchString), Logger::INFO);

    srand(static_cast <unsigned> (time(0)));

    requestInput("Select sample rate", sampleRate);
    classifier.initalize(sampleRate);
    logger.log(std::format("Set sample rate: {}", sampleRate), Logger::VERBOSE);

    {
        double forwardSamples = FFT_FRAME_SAMPLES + (CONTEXT_FORWARD * FFT_FRAME_SPACING);
        double backwardSamples = FFT_FRAME_SAMPLES + (CONTEXT_BACKWARD * FFT_FRAME_SPACING);
        double forwardMsec = 1000.0 * (forwardSamples / sampleRate);
        double backwardMsec = 1000.0 * (backwardSamples / sampleRate);
        logger.log(std::format("Forward context: {}ms; Backward context: {}ms", forwardMsec, backwardMsec), Logger::INFO);
    }

    float gain = 1;
    requestInput("Set gain", gain);
    logger.log(std::format("Set gain: {}", gain), Logger::VERBOSE);
    ClassifierHelper::instance().setGain(gain);
}

int main(int argc, char* argv[]) {
    tryMakeDir("logs");
    tryMakeDir("configs/articulators");
    tryMakeDir("configs/animations/phonemes");

    logger = Logger();
    logger.addStream(Logger::Stream("main.log")
        .outputTo(Logger::VERBOSE)
        .outputTo(Logger::INFO)
        .outputTo(Logger::WARNING)
        .outputTo(Logger::ERR)
        .outputTo(Logger::FATAL));
    logger.addStream(Logger::Stream(std::cout)
        .outputTo(Logger::INFO)
        .outputTo(Logger::WARNING)
        .outputTo(Logger::ERR)
        .outputTo(Logger::FATAL)
        .enableColor(true));

    cag_option_context context;
    cag_option_init(&context, options, CAG_ARRAY_SIZE(options), argc, argv);
    bool trainMode = false, preprocessMode = false, helpMode = false, interactiveMode = false;
    std::string tVal, pVal, wVal, dVal, aVal, oVal, iiVal;
    while (cag_option_fetch(&context)) {
        switch (cag_option_get_identifier(&context)) {
        case 't': // Train mode
            trainMode = true;
            tVal = cag_option_get_value(&context);
            break;
        case 'p': // Preprocess mode
            preprocessMode = true;
            pVal = cag_option_get_value(&context);
            break;
        case 'h': // Help mode
            helpMode = true;
            break;
        case 'w': // Temporary work folder
            wVal = cag_option_get_value(&context);
            break;
        case 'd': // MFA Dictionary path
            dVal = cag_option_get_value(&context);
            break;
        case 'a': // Acoustic model path
            aVal = cag_option_get_value(&context);
            break;
        case 'o': // Output location
            oVal = cag_option_get_value(&context);
            break;
        case '\\':
            interactiveMode = true;
            iiVal = cag_option_get_value(&context);
            break;
        }
    }

    int error = 0;

    if (helpMode) {
        error = commandHelp();
    } else if (preprocessMode) {
        error = commandPreprocess(pVal, wVal, dVal, aVal, oVal);
    } else {
        initClassifier(argc, argv);
        if (trainMode) {
            error = commandTrain(tVal);
        } else if (interactiveMode) {
            error = commandInteractive(iiVal);
        } else {
            error = commandDefault();
        }
    }

    classifier.destroy();

    return error;
}
