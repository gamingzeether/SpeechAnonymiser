#include "define.hpp"

#include "common_inc.hpp"

#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <thread>
#include <filesystem>
#include <random>
#include <unordered_set>
#include <format>
#include <optional>
#include <cargs.h>
#include "Classifier/PhonemeClassifier.hpp"
#include "Classifier/Train/Dataset.hpp"
#include "Classifier/Train/TSVReader.hpp"
#include "Utils/ClassifierHelper.hpp"
#include "Utils/Global.hpp"
#include "Utils/TranslationMap.hpp"
#include "Utils/Util.hpp"
#include "SpeechEngine/SpeechEngineConcatenator.hpp"
#include "SpeechEngine/SpeechEngineFormant.hpp"
#include "structs.hpp"
#ifdef AUDIO
#include <rtaudio/RtAudio.h>
#endif
#ifdef GUI
#include "Visualizer.hpp"
#endif

const bool outputPassthrough = false;

auto programStart = std::chrono::system_clock::now();
int sampleRate = 16000;

PhonemeClassifier classifier;

std::unique_ptr<SpeechEngine> speechEngine = nullptr;

static struct cag_option options[] = {
  {
    .identifier = 't',
    .access_letters = "t",
    .access_name = "train",
    .value_name = "[Dataset directory]",
    .description = "Phoneme classifier training mode"},

  {
    .identifier = 'p',
    .access_letters = "p",
    .access_name = "preprocess",
    .value_name = "[Common Voice directory]",
    .description = "Preprocess training data"},

  {
    .identifier = 'h',
    .access_letters = "h",
    .access_name = "help",
    .description = "Shows the command help"},

  {
    .identifier = 'w',
    .access_letters = "w",
    .value_name = "[Directory]",
    .description = "Work directory"},

  {
    .identifier = 'd',
    .access_letters = "d",
    .value_name = "[Dictionary]",
    .description = "MFA dictionary path"},

  {
    .identifier = 'a',
    .access_letters = "a",
    .value_name = "[Model]",
    .description = "MFA acoustic model path"},

  {
    .identifier = 'o',
    .access_letters = "o",
    .value_name = "[Directory]",
    .description = "Output directory"},

  {
    .identifier = '\\',
    .access_name = "interactive",
    .value_name = "[Dataset directory]",
    .description = "Used for development"},

   {
    .identifier = 'e',
    .access_letters = "e",
    .value_name = "[Dataset directory]",
    .description = "Evaluate model accuracy"}
};

struct AudioContainer {
    std::vector<float> audio;
    size_t pointer;
    std::mutex mtx;
};

template <typename T>
bool requestString(const std::string& request, std::string& out, T& value) {
    std::cout << request << std::endl << "(Default " << value << ") ";
    bool isInteractive = getenv("NONINTERACTIVE") == NULL;
    if (isInteractive) {
        std::getline(std::cin, out);
    }
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

#ifdef AUDIO
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

    if (outputPassthrough) {
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
    } else if (speechEngine) {
        speechEngine->writeBuffer((OUTPUT_TYPE*)outputBuffer, nBufferFrames);
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
        float data = (ptr < size) ? container.audio[ptr++] : 0;
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
#endif

#ifdef GUI
void startFFT(InputData& inputData) {
    float activationThreshold = 0.01;
    if (classifier.ready) {
        G_LG("Model successfully loaded", Logger::INFO);
        requestInput("Set activation threshold", activationThreshold);
    } else {
        G_LG("Model could not be loaded, disabling classification", Logger::WARN);
        //activationThreshold = std::numeric_limits<float>::max();
    }

    Visualizer app;
    G_LG("Starting visualizer", Logger::INFO);
    const size_t frameCount = FFT_FRAMES * 2;
    app.fftData.frames = frameCount;
    app.fftData.currentFrame = 0;
    app.fftData.frequencies = new float* [frameCount];
    for (size_t i = 0; i < frameCount; i++) {
        app.fftData.frequencies[i] = new float[FFT_REAL_SAMPLES];
        for (size_t j = 0; j < FFT_REAL_SAMPLES; j++) {
            app.fftData.frequencies[i][j] = 0.0;
        }
    }
    std::thread fft = std::thread([&app, &inputData, &activationThreshold] {
        // Setup classifier
        CPU_CUBE_TYPE data(classifier.getInputSize(), 1, 1);

        // Wait for visualization to open
        while (!app.isOpen) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        G_LG("Starting FFT thread processing", Logger::DBUG);
        std::vector<Frame> frames = std::vector<Frame>(frameCount);
        for (Frame& f : frames) {
            f.reset();
        }
        size_t currentFrame = 0;
        size_t lastSampleStart = 0;

        //std::printf("Maximum time per frame: %fms\n", (1000.0 * FFT_FRAME_SPACING) / classifier.getSampleRate());

        ClassifierHelper helper;
        app.classifierHelper = &helper;
        helper.initalize(sampleRate);
        SpeechFrame speechFrame;
        const size_t silencePhoneme = G_P_SIL;
        speechFrame.phoneme = silencePhoneme;
        int count = 0;
        // Wait for enough samples to be recorded to pass to FFT
        while ((inputData.writeOffset - lastSampleStart) % inputData.totalFrames < FFT_REAL_SAMPLES) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        TranslationMap tm(G_PS_C, G_PS_S);

        while (app.isOpen) {
            Frame& frame = frames[currentFrame];
            // Wait for FFT_FRAME_SPACING new samples
            //auto start = std::chrono::high_resolution_clock::now();
            while ((inputData.totalFrames + inputData.writeOffset - lastSampleStart) % inputData.totalFrames < FFT_FRAME_SPACING) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            //auto actual = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
            //std::cout << actual.count() << '\n';
            lastSampleStart = (lastSampleStart + FFT_FRAME_SPACING) % inputData.totalFrames;
            // Do FFT stuff
            helper.processFrame(inputData.buffer[0], lastSampleStart, inputData.totalFrames, frames, currentFrame);

            // Write FFT output to visualizer
            memcpy(app.fftData.frequencies[currentFrame], frame.avg.data(), sizeof(float) * FRAME_SIZE);
            app.fftData.currentFrame = currentFrame;
            app.updateWaterfall();

            // Check if volume is greater than the activation threshold
            bool activity = false;
            for (int i = 0; i < ACTIVITY_WIDTH; i++) {
                Frame& activityFrame = frames[(currentFrame + frameCount - i) % frameCount];
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
                    if (helper.writeInput(frames, currentFrame, data, 0, 0)) {
                        //auto classifyStart = std::chrono::high_resolution_clock::now();
                        size_t phoneme = classifier.classify(data);
                        speechFrame.phoneme = phoneme;
                        //auto classifyDuration = std::chrono::high_resolution_clock::now() - classifyStart;

                        std::cout << classifier.getPhonemeString(phoneme) << std::endl;
                        //std::cout << std::chrono::duration<double>(classifyDuration).count() * 1000 << " ms\n";
                    } else {
                        std::cout << "Error writing input\n";
                    }
                }
            } else {
                speechFrame.phoneme = silencePhoneme;
            }
            // Translate classifier -> speech engine and push
            speechFrame.phoneme = tm.translate(speechFrame.phoneme);
            if (speechEngine)
                speechEngine->pushFrame(speechFrame);
            currentFrame = (currentFrame + 1) % frameCount;
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
#endif

int commandHelp() {
    printf("Usage: SpeechAnonymiser [OPTION]...\n\n");
    cag_option_print(options, CAG_ARRAY_SIZE(options), stdout);
    return 0;
}

int commandTrain(const std::string& path) {
    G_LG("Training mode", Logger::INFO);

    int examples = 1000;
    requestInput("Set examples", examples);
    G_LG(std::format("Set examples count: {}", examples), Logger::DBUG);
    if (examples <= 0) {
        throw("Out of range");
    }

    int epochs = 1000;
    requestInput("Set number of epochs", epochs);
    G_LG(std::format("Set epochs: {}", epochs), Logger::DBUG);
    if (epochs <= 0) {
        throw("Out of range");
    }

    //double stepSize = STEP_SIZE;
    //requestInput("Set training rate", stepSize);
    //G_LG(std::format("Set training rate: {}", stepSize), Logger::DBUG);
    //if (stepSize <= 0) {
    //    throw("Out of range");
    //}

    classifier.train(path, examples, epochs);

    return 0;
}

int commandPreprocess(const std::string& path, const std::string& workDir, const std::string& dictPath, const std::string& acousticPath, const std::string& outputDir) {
    char* condaEnv = getenv("CONDA_DEFAULT_ENV");
    if (condaEnv == NULL || strcmp(condaEnv, "aligner") != 0) {
        std::string errorMessage = "Aligner not detected, make sure MFA is installed and activated before starting this program";
        G_LG(errorMessage, Logger::WARN);
    }

    int batchSize = 1500;
    requestInput("Set batch size: ", batchSize);
    
    Dataset ds = Dataset();
    ds.preprocessDataset(path, workDir, dictPath, acousticPath, outputDir, batchSize);

    return 0;
}

#ifdef AUDIO
  #ifdef GUI
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

    G_LG(std::format("Using input: {}", inputInfo.name), Logger::INFO);
    G_LG(std::format("Sample rate: {}", inputSampleRate), Logger::INFO);
    G_LG(std::format("Channels: {}", inputInfo.inputChannels), Logger::INFO);

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


    G_LG(std::format("Using output: {}", outputInfo.name), Logger::INFO);
    G_LG(std::format("Sample rate: {}", outputSampleRate), Logger::INFO);
    G_LG(std::format("Channels: {}", outputInfo.outputChannels), Logger::INFO);

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
    /*
    auto tmpEngine = SpeechEngineConcatenator();
    tmpEngine.setSampleRate(outputSampleRate)
        .setChannels(2)
        .setVolume(0.05)
        .configure("AERIS CV-VC ENG Kire 2.0/");
    */
    /*
    auto tmpEngine = SpeechEngineFormant();
    tmpEngine.setSampleRate(outputSampleRate)
        .setChannels(2)
        .setVolume(0.001)
        .configure("configs/formants/formants.json");
    speechEngine = std::unique_ptr<SpeechEngine>(&tmpEngine);
    */

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
  #endif

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


    G_LG(std::format("Using output: {}", outputInfo.name), Logger::INFO);
    G_LG(std::format("Sample rate: {}", outputSampleRate), Logger::INFO);
    G_LG(std::format("Channels: {}", outputInfo.outputChannels), Logger::INFO);

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

    Dataset ds = Dataset(outputSampleRate, path);
    ds.setSubtype(Subtype::TEST);

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
        std::printf("%s: ", prefix.c_str());
        std::getline(std::cin, command);
        if (command == "voicebank" && prefix == "~") {
            prefix = "voicebank";
        } else if (command == "dataset" && prefix == "~") {
            prefix = "dataset";
        } else if (command == "phones" && prefix == "~") {
            for (int i = 0; i < G_PS_C.size(); i++) {
                std::string xs = G_PS_C.xSampa(i);
                std::printf("%2d : %s\n", i, xs.c_str());
            }
        } else if (command == "exit") {
            if (prefix == "clip") {
                prefix = "dataset";
            } else {
                prefix = "~";
            }
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
            if (targetPhoneme < G_PS_C.size()) {
                std::string fileName;
                clipAudio = ds._findAndLoad(path, targetPhoneme, outputSampleRate, fileName, clipPhones);
                prefix = "clip";
                std::printf("%s\n", fileName.c_str());
                for (int i = 0; i < clipPhones.size(); i++) {
                    const Phone& p = clipPhones[i];
                    std::printf("%2d  %s: %.2f, %.2f\n", i, G_PS_C.xSampa(p.phonetic).c_str(), p.min, p.max);
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
#endif

int commandEvaluate(const std::string& path) {
    classifier.evaluate(path);
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

void initClassifier() {
    srand(static_cast <unsigned> (time(0)));

    //requestInput("Select sample rate", sampleRate);
    sampleRate = 16000;
    classifier.initalize(sampleRate);
    G_LG(std::format("Set sample rate: {}", sampleRate), Logger::DBUG);

    {
        double forwardSamples = FFT_FRAME_SAMPLES + (CONTEXT_FORWARD * FFT_FRAME_SPACING);
        double backwardSamples = FFT_FRAME_SAMPLES + (CONTEXT_BACKWARD * FFT_FRAME_SPACING);
        double forwardMsec = 1000.0 * (forwardSamples / sampleRate);
        double backwardMsec = 1000.0 * (backwardSamples / sampleRate);
        G_LG(std::format("Forward context: {}ms; Backward context: {}ms", forwardMsec, backwardMsec), Logger::INFO);
    }
}

void setClassifierSet(const std::string& datasetPath) {
    // Set classifer phoneme set
    Type datasetType = Dataset::folderType(datasetPath);
    switch (datasetType) {
        case Type::COMMON_VOICE:
            Global::get().setClassifierPhonemeSet("CVClassifier");
            break;
        case Type::TIMIT:
            Global::get().setClassifierPhonemeSet("TIMITClassifier");
            break;
    }
}

int main(int argc, char* argv[]) {
    // Can't print to cout if I don't write something to it here???
    std::cout << "\r";

    std::string launchString = "";
    for (int i = 0; i < argc; i++) {
        if (i > 0) {
            launchString += " ";
        }
        launchString += argv[i];
    }
    G_LG(std::format("Launch args: {}", launchString), Logger::INFO);
    G_LG(std::format("Working directory: {}", std::filesystem::current_path().string()), Logger::INFO);

    tryMakeDir("logs");
    tryMakeDir("configs/articulators");
    tryMakeDir("configs/animations/phonemes");

    cag_option_context context;
    cag_option_init(&context, options, CAG_ARRAY_SIZE(options), argc, argv);
    bool trainMode = false, preprocessMode = false, helpMode = false, interactiveMode = false, evaluateMode = false;
    std::string tVal, pVal, wVal, dVal, aVal, oVal, iiVal, eVal;
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
        case 'e':
            evaluateMode = true;
            eVal = cag_option_get_value(&context);
            break;
        }
    }

    int error = 0;

    if (helpMode) {
        error = commandHelp();
    } else if (preprocessMode) {
        Util::removeTrailingSlash(pVal);
        Util::removeTrailingSlash(wVal);
        Util::removeTrailingSlash(dVal);
        Util::removeTrailingSlash(aVal);
        Util::removeTrailingSlash(oVal);
        error = commandPreprocess(pVal, wVal, dVal, aVal, oVal);
    } else {
        setClassifierSet(Util::firstNotOf<std::string>({tVal, iiVal, eVal}, ""));
        initClassifier();
        if (trainMode) {
            Util::removeTrailingSlash(tVal);
            error = commandTrain(tVal);
        } else if (interactiveMode) {
#ifdef AUDIO
            Util::removeTrailingSlash(iiVal);
            error = commandInteractive(iiVal);
#else
            G_LG("Compiled without audio support, exiting", Logger::DEAD);
            error = -1;
#endif
        } else if (evaluateMode) {
            Util::removeTrailingSlash(eVal);
            error = commandEvaluate(eVal);
        } else {
#ifdef AUDIO
  #ifdef GUI
            error = commandDefault();
  #else
            G_LG("Compiled without GUI support, exiting", Logger::DEAD);
            error = -1;
  #endif
#else
            G_LG("Compiled without audio support, exiting", Logger::DEAD);
            error = -1;
#endif
        }
    }

    classifier.destroy();

    G_LG(std::format("Program exited with code {}", error), Logger::INFO);

    return error;
}
