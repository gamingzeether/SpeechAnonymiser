﻿#include "define.h"

#include "common_inc.h"

#define DR_MP3_IMPLEMENTATION
#define DR_WAV_IMPLEMENTATION
#define MLPACK_ENABLE_ANN_SERIALIZATION
#define ARMA_PRINT_EXCEPTIONS

#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <thread>
#include <filesystem>
#include <random>
#include <unordered_set>
#include <filesystem>
#include <rtaudio/RtAudio.h>
#include <fftw3.h>
#include <cargs.h>
#include <samplerate.h>
#include <dr_mp3.h>
#include <dr_wav.h>
#include <mlpack/mlpack.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/rnn.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann.hpp>
#include "Visualizer.h"
#include "TSVReader.h"

using namespace arma;
using namespace mlpack;

float* window;
float* fftwIn;
fftwf_complex* fftwOut;
fftwf_plan fftwPlan;

std::unordered_map<size_t, size_t> phonemeSet;
std::vector<std::wstring> inversePhonemeSet;

auto programStart = std::chrono::system_clock::now();
int SAMPLE_RATE;

size_t inputSize = FFT_REAL_SAMPLES;
size_t outputSize = 0;
size_t hiddenSize = FFT_REAL_SAMPLES;

std::string modelFile = "phoneme_rnn";
std::string modelExt = ".bin";

ann::RNN<MeanSquaredError, HeInitialization> network;
ens::Adam optimizer;

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
        buffer = new float*[ch];
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

struct Clip {
    std::string clipPath;
    std::string* tsvElements;
    float* buffer;
    size_t size;
    std::string sentence;
    unsigned int sampleRate;
    bool loaded = false;

    void loadMP3(int targetSampleRate) {
        // Read mp3 file from tsv
#pragma omp critical
        std::cout << "Loading MP3: " << tsvElements[TSVReader::Indices::PATH] << '\n';

        drmp3_config cfg;
        drmp3_uint64 samples;

        std::string clipFullPath = clipPath + tsvElements[TSVReader::Indices::PATH];
        if (!std::filesystem::exists(clipFullPath)) {
#pragma omp critical
            std::cout << clipFullPath << " does not exist\n";
            return;
        }

        float* floatBuffer = drmp3_open_file_and_read_pcm_frames_f32(clipFullPath.c_str(), &cfg, &samples, NULL);

        if (cfg.channels <= 0) {
#pragma omp critical
            std::cout << clipFullPath << " has invalid channel count (" << cfg.channels << ")\n";
            return;
        }
        if (cfg.sampleRate <= 0) {
#pragma omp critical
            std::cout << clipFullPath << " has invalid sample rate (" << cfg.sampleRate << ")\n";
            return;
        }

        if (cfg.sampleRate == targetSampleRate) {
            memcpy(buffer, floatBuffer, sizeof(float) * samples);
            size = samples;
            sampleRate = cfg.sampleRate;
        } else {
            SRC_DATA upsampleData = SRC_DATA();
            upsampleData.data_in = floatBuffer;
            upsampleData.input_frames = samples;
            double ratio = (double)targetSampleRate / cfg.sampleRate;
            long outSize = (long)(ratio * samples);
            upsampleData.src_ratio = ratio;
            upsampleData.data_out = buffer;
            upsampleData.output_frames = outSize;
            int error = src_simple(&upsampleData, SRC_SINC_BEST_QUALITY, cfg.channels);
            if (error) {
                std::cout << "Error while upsampling: " << src_strerror(error) << '\n';
            }
            sampleRate = targetSampleRate;
            size = outSize;
        }
        free(floatBuffer);
        loaded = true;
    }

    Clip() {
        clipPath = "";
        tsvElements = NULL;
        buffer = new float[SAMPLE_RATE * CLIP_LENGTH];
        size = 0;
        sentence = "";
        sampleRate = 0;
    }
    ~Clip() {
        delete[] tsvElements;
        delete[] buffer;
    }
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
    std::vector<float> imaginary;
    float volume;

    void reset() {
        for (size_t i = 0; i < FFT_REAL_SAMPLES; i++) {
            real[i] = 0;
            imaginary[i] = 0;
        }
        volume = 0;
    }

    Frame() {
        real = std::vector<float>(FFT_REAL_SAMPLES);
        imaginary = std::vector<float>(FFT_REAL_SAMPLES);
        volume = 0;
    }
};

// https://stackoverflow.com/a/7154226
std::wstring utf8_to_utf16(const std::string& utf8)
{
    std::vector<unsigned long> unicode;
    size_t i = 0;
    while (i < utf8.size())
    {
        unsigned long uni;
        size_t todo;
        bool error = false;
        unsigned char ch = utf8[i++];
        if (ch <= 0x7F)
        {
            uni = ch;
            todo = 0;
        } else if (ch <= 0xBF)
        {
            throw std::logic_error("not a UTF-8 string");
        } else if (ch <= 0xDF)
        {
            uni = ch & 0x1F;
            todo = 1;
        } else if (ch <= 0xEF)
        {
            uni = ch & 0x0F;
            todo = 2;
        } else if (ch <= 0xF7)
        {
            uni = ch & 0x07;
            todo = 3;
        } else
        {
            throw std::logic_error("not a UTF-8 string");
        }
        for (size_t j = 0; j < todo; ++j)
        {
            if (i == utf8.size())
                throw std::logic_error("not a UTF-8 string");
            unsigned char ch = utf8[i++];
            if (ch < 0x80 || ch > 0xBF)
                throw std::logic_error("not a UTF-8 string");
            uni <<= 6;
            uni += ch & 0x3F;
        }
        if (uni >= 0xD800 && uni <= 0xDFFF)
            throw std::logic_error("not a UTF-8 string");
        if (uni > 0x10FFFF)
            throw std::logic_error("not a UTF-8 string");
        unicode.push_back(uni);
    }
    std::wstring utf16;
    for (size_t i = 0; i < unicode.size(); ++i)
    {
        unsigned long uni = unicode[i];
        if (uni <= 0xFFFF)
        {
            utf16 += (wchar_t)uni;
        } else
        {
            uni -= 0x10000;
            utf16 += (wchar_t)((uni >> 10) + 0xD800);
            utf16 += (wchar_t)((uni & 0x3FF) + 0xDC00);
        }
    }
    return utf16;
}

size_t customHasher(const std::wstring& str) {
    size_t v = 0;
    for (size_t i = 0; i < str.length(); i++) {
        v = (v << sizeof(wchar_t) * 8) ^ str[i];
    }
    return v;
}

void initalizePhonemeSet() {
#define REGISTER_PHONEME(p) \
    phonemeSet[customHasher(L##p)] = _phonemeCounter++; \
    inversePhonemeSet.push_back(L##p);

    size_t _phonemeCounter = 0;
    using namespace std::string_literals;
    REGISTER_PHONEME(""   )
    REGISTER_PHONEME("aj" )
    REGISTER_PHONEME("aw" )
    REGISTER_PHONEME("b"  )
    REGISTER_PHONEME("bʲ" )
    REGISTER_PHONEME("c"  )
    REGISTER_PHONEME("cʰ" )
    REGISTER_PHONEME("cʷ" )
    REGISTER_PHONEME("d"  )
    REGISTER_PHONEME("dʒ" )
    REGISTER_PHONEME("dʲ" )
    REGISTER_PHONEME("d̪" )
    REGISTER_PHONEME("ej" )
    REGISTER_PHONEME("f"  )
    REGISTER_PHONEME("fʲ" )
    REGISTER_PHONEME("h"  )
    REGISTER_PHONEME("i"  )
    REGISTER_PHONEME("iː" )
    REGISTER_PHONEME("j"  )
    REGISTER_PHONEME("k"  )
    REGISTER_PHONEME("kʰ" )
    REGISTER_PHONEME("kʷ" )
    REGISTER_PHONEME("l"  )
    REGISTER_PHONEME("m"  )
    REGISTER_PHONEME("mʲ" )
    REGISTER_PHONEME("m̩" )
    REGISTER_PHONEME("n"  )
    REGISTER_PHONEME("n̩" )
    REGISTER_PHONEME("ow" )
    REGISTER_PHONEME("p"  )
    REGISTER_PHONEME("pʰ" )
    REGISTER_PHONEME("pʲ" )
    REGISTER_PHONEME("pʷ" )
    REGISTER_PHONEME("s"  )
    REGISTER_PHONEME("spn")
    REGISTER_PHONEME("t"  )
    REGISTER_PHONEME("tʃ" )
    REGISTER_PHONEME("tʰ" )
    REGISTER_PHONEME("tʲ" )
    REGISTER_PHONEME("tʷ" )
    REGISTER_PHONEME("t̪" )
    REGISTER_PHONEME("v"  )
    REGISTER_PHONEME("vʲ" )
    REGISTER_PHONEME("w"  )
    REGISTER_PHONEME("z"  )
    REGISTER_PHONEME("æ"  )
    REGISTER_PHONEME("ç"  )
    REGISTER_PHONEME("ð"  )
    REGISTER_PHONEME("ŋ"  )
    REGISTER_PHONEME("ɐ"  )
    REGISTER_PHONEME("ɑ"  )
    REGISTER_PHONEME("ɑː" )
    REGISTER_PHONEME("ɒ"  )
    REGISTER_PHONEME("ɒː" )
    REGISTER_PHONEME("ɔj" )
    REGISTER_PHONEME("ə"  )
    REGISTER_PHONEME("ɚ"  )
    REGISTER_PHONEME("ɛ"  )
    REGISTER_PHONEME("ɝ"  )
    REGISTER_PHONEME("ɟ"  )
    REGISTER_PHONEME("ɟʷ" )
    REGISTER_PHONEME("ɡ"  )
    REGISTER_PHONEME("ɡʷ" )
    REGISTER_PHONEME("ɪ"  )
    REGISTER_PHONEME("ɫ"  )
    REGISTER_PHONEME("ɫ̩" )
    REGISTER_PHONEME("ɱ"  )
    REGISTER_PHONEME("ɲ"  )
    REGISTER_PHONEME("ɹ"  )
    REGISTER_PHONEME("ɾ"  )
    REGISTER_PHONEME("ɾʲ" )
    REGISTER_PHONEME("ɾ̃"  )
    REGISTER_PHONEME("ʃ"  )
    REGISTER_PHONEME("ʉ"  )
    REGISTER_PHONEME("ʉː" )
    REGISTER_PHONEME("ʊ"  )
    REGISTER_PHONEME("ʎ"  )
    REGISTER_PHONEME("ʒ"  )
    REGISTER_PHONEME("ʔ"  )
    REGISTER_PHONEME("θ"  )
#undef REGISTER_PHONEME
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

void processFrame(Frame& frame, const float* audio, const size_t& start, const size_t& totalSize) {
    float max = 0.0;
    for (size_t i = 0; i < FFT_FRAME_SAMPLES; i++) {
        size_t readLocation = (start + i) % totalSize;
        const float& value = audio[readLocation];
        fftwIn[i] = value * window[i];
        max = fmaxf(max, abs(value));
    }
    frame.volume = max;
    fftwf_execute(fftwPlan);
    for (size_t i = 0; i < FFT_REAL_SAMPLES; i++) {
        fftwf_complex& complex = fftwOut[i];
        frame.real[i] = abs(complex[0]);
        frame.imaginary[i] = abs(complex[1]);
    }
}

void startFFT(InputData& inputData) {
    Visualizer app;
    app.initWindow();
    app.fftData.frames = FFT_FRAMES;
    app.fftData.currentFrame = 0;
    app.fftData.frequencies = new float* [FFT_FRAMES];
    for (size_t i = 0; i < FFT_FRAMES; i++) {
        app.fftData.frequencies[i] = new float[FFT_REAL_SAMPLES];
        for (size_t j = 0; j < FFT_REAL_SAMPLES; j++) {
            app.fftData.frequencies[i][j] = 0.0;
        }
    }
    std::thread fft = std::thread([&app, &inputData] {
        // Setup classifier
        cube data(inputSize, 1, FFT_FRAMES);
        cube out(outputSize, 1, FFT_FRAMES);
        bool loaded = data::Load(modelFile + modelExt, "model", network, true);

        // Wait for visualization to open
        while (!app.isOpen) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        Frame frame = Frame();
        size_t lastSampleStart = 0;
        while (app.isOpen) {
            // Wait for enough samples to be recorded to pass to FFT
            while ((inputData.writeOffset - lastSampleStart) % inputData.totalFrames < FFT_FRAME_SAMPLES) {
                //auto start = std::chrono::high_resolution_clock::now();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                //auto actual = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
                //std::cout << actual.count() << '\n';
            }
            lastSampleStart = (lastSampleStart + FFT_FRAME_SPACING) % inputData.totalFrames;
            // Do FFT stuff
            processFrame(frame, inputData.buffer[0], lastSampleStart, inputData.totalFrames);

            // Write FFT output to visualizer
            memcpy(app.fftData.frequencies[app.fftData.currentFrame], frame.real.data(), sizeof(float) * FFT_REAL_SAMPLES);
            app.fftData.currentFrame = (app.fftData.currentFrame + 1) % FFT_FRAMES;

            // Pass data to neural network
            for (size_t i = 0; i < FFT_FRAMES; i++) {
                int frameIdx = (app.fftData.currentFrame + i) % FFT_FRAMES;
                float* frameData = app.fftData.frequencies[frameIdx];
                for (size_t j = 0; j < inputSize; j++) {
                    data(j, 0, i) = frameData[j];
                }
            }
            network.Predict(data, out);
            size_t predicted = 0;
            float max = out(0, 0, FFT_FRAMES - 1 - IDENTIFY_LATENCY);
            for (size_t i = 0; i < outputSize; i++) {
                const float& val = out(i, 0, FFT_FRAMES - 1 - IDENTIFY_LATENCY);
                if (val > max) {
                    max = val;
                    predicted = i;
                }
            }
            std::cout << predicted << std::endl;
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

char* combine(const char* a, const char* b) {
    int aLength = 0, bLength = 0;
    while (a[aLength] != NULL) {
        aLength++;
    }
    while (b[bLength] != NULL) {
        bLength++;
    }
    char* combined = new char[aLength + bLength + 1];
    for (int i = 0; i < aLength; i++) {
        combined[i] = a[i];
    }
    for (int i = 0; i < bLength; i++) {
        combined[aLength + i] = b[i];
    }
    combined[aLength + bLength] = NULL;
    return combined;
}

void loadNextClip(const std::string& clipPath, TSVReader& tsv, OUT Clip& clip, int sampleRate) {
    clip.clipPath = clipPath;
    clip.loaded = false;
    std::string* elements = tsv.read_line();
    clip.tsvElements = elements;
    if (sampleRate > 0) {
        clip.loadMP3(sampleRate);
    }
}

int commandHelp() {
    printf("Usage: SpeechAnonymiser [OPTION]...\n\n");
    cag_option_print(options, CAG_ARRAY_SIZE(options), stdout);
    return 0;
}

int commandTrain(const char* path) {
    //InputData inputData = InputData(2, SAMPLE_RATE * INPUT_BUFFER_TIME);

    // Create and start output device
#pragma region Output
    /*
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
    */
#pragma endregion

    std::vector<Phone> phones;
    std::cout << "Training mode\n";

    TSVReader tsv;
    tsv.open(combine(path, "/train.tsv"));
    const std::string clipPath = std::string(combine(path, "/clips/"));
    const std::string transcriptsPath = std::string(combine(path, "/transcript/"));

    std::vector<Frame> frames;

    size_t batchSize = 100;
    std::cout << "Set batch size (Default: " << batchSize << ")\n";
    std::string response;
    std::getline(std::cin, response);
    if (response != "") {
        batchSize = std::stoi(response); // Exits if unparsable
        if (batchSize <= 0) {
            assert("Out of range");
        }
    }

    std::vector<Clip> clips = std::vector<Clip>(batchSize);
    std::vector<std::thread> loaders = std::vector<std::thread>(batchSize);

    bool loaded = data::Load(modelFile + modelExt, "model", network);
    if (loaded) {
        std::cout << "Loaded model\n";
    } else {
        std::cout << "No model found, training new model\n";

        network.Add<Linear>(inputSize);
        network.Add<LeakyReLU>();
        network.Add<LSTM>(inputSize / 2);
        network.Add<LeakyReLU>();
        network.Add<Linear>(outputSize);
        network.Add<LogSoftMax>();
    }

    optimizer = ens::Adam(
        STEP_SIZE,  // Step size of the optimizer.
        batchSize, // Batch size. Number of data points that are used in each iteration.
        0.9,        // Exponential decay rate for the first moment estimates.
        0.999, // Exponential decay rate for the weighted infinity norm estimates.
        1e-8,  // Value used to initialise the mean squared gradient parameter.
        EPOCHS * batchSize, // Max number of iterations.
        1e-8,           // Tolerance.
        true);

    while (tsv.good()) {
        size_t clipCount = 0;
        for (size_t i = 0; i < batchSize && tsv.good(); i++) {
            loaders[i] = std::thread(loadNextClip, std::ref(clipPath), std::ref(tsv), std::ref(clips[i]), -1);
            clipCount++;
        }
#pragma omp parallel for
        for (size_t i = 0; i < clipCount; i++) {
            clips[i].loadMP3(SAMPLE_RATE);
        }
        size_t maxFrames = 0;
        size_t actualLoadedClips = 0;
        for (size_t i = 0; i < clipCount; i++) {
            if (clips[i].loaded) {
                maxFrames = std::max(maxFrames, (clips[i].size + FFT_FRAME_SAMPLES) / FFT_FRAME_SPACING);
                actualLoadedClips++;
            }
        }
        std::cout << actualLoadedClips << " clips loaded out of " << clipCount << " total\n";
        std::cout << maxFrames << " frames\n";

        arma::cube train = cube(inputSize, clipCount, maxFrames);
        arma::cube labels = cube(outputSize, clipCount, maxFrames);
        train.fill(0.0);
        labels.fill(-1.0);

#pragma omp parallel for
        for (intptr_t c = 0; c < clipCount; c++) {
            Clip& currentClip = clips[c];
            if (!currentClip.loaded) {
                continue;
            }

#pragma region Really shitty textgrid parser
            // Get transcription
            const std::string path = currentClip.tsvElements[TSVReader::Indices::PATH];
            const std::string transcriptionPath = transcriptsPath + currentClip.tsvElements[TSVReader::Indices::CLIENT_ID] + "/" + path.substr(0, path.length() - 4) + ".TextGrid";
            if (!std::filesystem::exists(transcriptionPath)) {
                continue;
            }
            std::ifstream reader;
            reader.open(transcriptionPath);

            std::string line;
            while (reader.good()) {
                std::getline(reader, line);
                if (line == "        name = \"phones\" ") {
                    for (int i = 0; i < 2; i++) {
                        std::getline(reader, line);
                    }
                    break;
                }
            }
            std::getline(reader, line);
            std::string sizeString = line.substr(line.find_last_of(' ', line.length() - 2));
            int size = std::stoi(sizeString);
            phones = std::vector<Phone>(size);
            std::string interval, xmin, xmax, text;
            for (int i = 0; i < size; i++) {
                std::getline(reader, interval);
                std::getline(reader, xmin);
                std::getline(reader, xmax);
                std::getline(reader, text);

                xmin = xmin.substr(xmin.find_last_of(' ', xmin.length() - 2));
                xmax = xmax.substr(xmax.find_last_of(' ', xmax.length() - 2));
                size_t textStart = text.find_last_of('\"', text.length() - 3) + 1;
                text = text.substr(textStart, text.length() - textStart - 2);

                Phone p = Phone();
                p.min = std::stod(xmin);
                p.max = std::stod(xmax);
                p.minIdx = SAMPLE_RATE * p.min;
                p.maxIdx = SAMPLE_RATE * p.max;
                
                std::wstring wtext = utf8_to_utf16(text);
                
                const auto& pIterator = phonemeSet.find(customHasher(wtext));
                if (pIterator != phonemeSet.end()) {
                    p.phonetic = pIterator->second;
                } else {
                    std::cout << "unmapped phoneme: " << text << std::endl;
                    std::cout << "    at index " << i << " in " << transcriptionPath << std::endl;
                }
                phones[i] = p;
            }

            reader.close();
#pragma endregion

            // Process frames of the whole clip
            size_t fftStart = 0;
            size_t currentFrame = 0;

            while ((size_t)fftStart + FFT_FRAME_SAMPLES < currentClip.size) {
                if (frames.size() <= currentFrame) {
                    frames.push_back(Frame());
                }
                Frame& frame = frames[currentFrame];
                frame.reset();

                processFrame(frame, currentClip.buffer, fftStart, currentClip.size);

                size_t maxOverlap = 0;
                size_t maxIdx = 0;
                for (int i = 0; i < size; i++) {
                    const Phone& p = phones[i];

                    if (p.maxIdx <= fftStart)
                        continue;
                    size_t fftEnd = fftStart + FFT_FRAME_SPACING;
                    if (p.minIdx >= fftEnd)
                        break;

                    size_t overlapA = p.maxIdx - fftStart;
                    size_t overlapB = fftEnd - p.minIdx;
                    size_t overlapC = FFT_FRAME_SPACING; // Window size
                    size_t overlapSize = std::min(std::min(overlapA, overlapB), overlapC);

                    if (overlapSize > maxOverlap) {
                        overlapSize = maxOverlap;
                        maxIdx = p.phonetic;
                    }
                }

                for (size_t i = 0; i < inputSize; i++) {
                    train(i, c, currentFrame) = frame.real[i];
                }
                labels(maxIdx, c, currentFrame) = 0.0;

                fftStart += FFT_FRAME_SPACING;
                currentFrame++;
            }

            // Label remaining frames at 0
            for (size_t i = currentFrame; i < maxFrames; i++) {
                labels(0, c, i) = 0.0;
            }
        }

        network.BPTTSteps() = maxFrames;
        network.Train(train,
            labels,
            optimizer,
            ens::PrintLoss(),
            ens::ProgressBar(),
            ens::EarlyStopAtMinLoss());

        data::Save(modelFile + modelExt, "model", network, true);
    }

    //cleanupRtAudio(outputAudio);

    return 0;
}

int commandPreprocess(const char* path) {
    // Generate phoneme list
    TSVReader dictReader;
    dictReader.open(combine(path, "/english_us_mfa.dict"));
    std::vector<std::string> phonemeList = std::vector<std::string>();
    while (dictReader.good()) {
        std::string* tabSeperated = dictReader.read_line();
        std::string& phonemes = tabSeperated[1];
        int start = 0;
        int end = 0;
        std::string p;
        while (end != -1) {
            end = phonemes.find_first_of(' ', start);
            p = phonemes.substr(start, end - start);
            start = end + 1;

            bool exists = false;
            for (size_t i = 0; i < phonemeList.size(); i++) {
                if (phonemeList[i] == p) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                phonemeList.push_back(p);
            }
        }
    }
    std::sort(phonemeList.begin(), phonemeList.end());
    std::fstream out = std::fstream("F:/Data/phonemes.txt", std::fstream::out);
    const char* newLine = "\n";
    for (size_t i = 0; i < phonemeList.size(); i++) {
        std::string& p = phonemeList[i];
        out.write(p.data(), p.size());
        out.write(newLine, 1);
    }
    out.close();

    // Generate phoneme alignments
    const std::vector<std::string> tables = {
        "/train.tsv",
        "/dev.tsv",
        "/test.tsv"
    };
    const std::string clipPath = std::string(combine(path, "/clips/"));
    const std::string corpusPath = "G:/corpus/";
    for (int i = 0; i < tables.size(); i++) {
        TSVReader tsv;
        tsv.open(combine(path, tables[i].c_str()));

        unsigned long globalCounter = 0;
        while (tsv.good()) {
            auto start = std::chrono::system_clock::now();
            // Batches
            int counter = 0;
            while (tsv.good() && !(counter >= PREPROCESS_BATCH_SIZE && globalCounter % PREPROCESS_BATCH_SIZE == 0)) {
                std::cout << globalCounter << "\n";

                Clip clip;
                loadNextClip(clipPath, tsv, clip, -1);

                globalCounter++;
                std::string transcriptPath = std::string(path) + "/transcript/" + clip.tsvElements[TSVReader::Indices::CLIENT_ID] + "/" + clip.tsvElements[TSVReader::Indices::PATH];
                transcriptPath = transcriptPath.substr(0, transcriptPath.length() - 4) + ".TextGrid";
                if (std::filesystem::exists(transcriptPath)) {
                    continue;
                }
                clip.loadMP3(16000);

                std::string speakerPath = corpusPath + clip.tsvElements[TSVReader::Indices::CLIENT_ID] + "/";
                if (!std::filesystem::is_directory(speakerPath)) {
                    std::filesystem::create_directory(speakerPath);
                }
                std::string originalPath = clip.tsvElements[TSVReader::Indices::PATH];
                std::string fileName = speakerPath + originalPath.substr(0, originalPath.length() - 4);

                // Convert audio to wav
                drwav wav;
                drwav_data_format format = drwav_data_format();
                format.container = drwav_container_riff;
                format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
                format.channels = 1;
                format.sampleRate = 16000;
                format.bitsPerSample = 32;
                drwav_init_file_write(&wav, (fileName + ".wav").c_str(), &format, NULL);
                drwav_uint64 framesWritten = drwav_write_pcm_frames(&wav, clip.size, clip.buffer);
                drwav_uninit(&wav);

                // Transcript
                std::ofstream fstream;
                fstream.open(fileName + ".txt");
                std::string sentence = clip.tsvElements[TSVReader::Indices::SENTENCE];
                fstream.write(sentence.c_str(), sentence.length());
                fstream.close();

                counter++;
            }
            // Run alignment
            // TODO: Change hardcoded paths
            system("conda activate aligner && \
              mfa align --clean \
                G:/corpus/ \
                F:/Data/cv-corpus-18.0-2024-06-14-en/en/english_us_mfa.dict \
                F:/Data/cv-corpus-18.0-2024-06-14-en/en/english_mfa.zip \
                F:/Data/cv-corpus-18.0-2024-06-14-en/en/transcript/");
            // Cleanup
            std::filesystem::directory_iterator iterator(corpusPath);
            for (const auto& directory : iterator) {
                std::filesystem::remove_all(directory);
            }
            auto diff = std::chrono::system_clock::now() - start;
            double duration = std::chrono::duration_cast<std::chrono::duration<double>>(diff).count();
            std::cout << "Iteration took " << duration << " seconds\n";
        }
    }

    return 0;
}

int commandDefault() {
#pragma region Input and output selector
    RtAudio audioQuery;
    auto devices = audioQuery.getDeviceIds();
    unsigned int selectedInput = audioQuery.getDefaultInputDevice();
    if (selectedInput == 0) {
        std::cout << "No input devices available\n";
        return 0;
    }
    unsigned int selectedOutput = audioQuery.getDefaultOutputDevice();
    if (selectedOutput == 0) {
        std::cout << "No output devices available\n";
        return 0;
    }
    std::vector<RtAudio::DeviceInfo> inputDevices = std::vector<RtAudio::DeviceInfo>();
    std::vector<RtAudio::DeviceInfo> outputDevices = std::vector<RtAudio::DeviceInfo>();
    size_t defaultInIdx = 0; // Just for display
    size_t defaultOutIdx = 0;
    for (size_t i = 0; i < devices.size(); i++) {
        unsigned int deviceId = devices[i];
        RtAudio::DeviceInfo deviceInfo = audioQuery.getDeviceInfo(deviceId);
        if (deviceInfo.inputChannels > 0) {
            if (deviceInfo.isDefaultInput) {
                defaultInIdx = inputDevices.size();
            }
            inputDevices.push_back(deviceInfo);
        }
        if (deviceInfo.outputChannels > 0) {
            if (deviceInfo.isDefaultOutput) {
                defaultOutIdx = outputDevices.size();
            }
            outputDevices.push_back(deviceInfo);
        }
    }

    std::string response;
    int responseInt;
    std::cout << "Select input device (default: " << defaultInIdx << ")\n";
    for (size_t i = 0; i < inputDevices.size(); i++) {
        std::cout << i << ": " << inputDevices[i].name << std::endl;
    }
    std::getline(std::cin, response);
    if (response != "") {
        responseInt = std::stoi(response); // Exit if unparsable
        if (responseInt < inputDevices.size()) {
            assert("Out of range");
        }
        selectedInput = inputDevices[responseInt].ID;
    }
    std::cout << "Select output device (default: " << defaultOutIdx << ")\n";
    for (size_t i = 0; i < outputDevices.size(); i++) {
        std::cout << i << ": " << outputDevices[i].name << std::endl;
    }
    std::getline(std::cin, response);
    if (response != "") {
        responseInt = std::stoi(response); // Exit if unparsable
        if (responseInt < outputDevices.size()) {
            assert("Out of range");
        }
        selectedOutput = outputDevices[responseInt].ID;
    }
#pragma endregion

    // Create and start input device
#pragma region Input
    RtAudio::StreamOptions inputFlags;
    inputFlags.flags |= RTAUDIO_NONINTERLEAVED;

    RtAudio inputAudio;

    RtAudio::StreamParameters inputParameters;

    RtAudio::DeviceInfo inputInfo = inputAudio.getDeviceInfo(selectedInput);

    inputParameters.deviceId = selectedInput;
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
    RtAudio::DeviceInfo outputInfo = outputAudio.getDeviceInfo(selectedOutput);

    outputParameters.deviceId = selectedOutput;
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

    // Setup data visualization
    startFFT(inputData);

    cleanupRtAudio(inputAudio);
    cleanupRtAudio(outputAudio);

    return 0;
}

int main(int argc, char** argv) {
    srand(static_cast <unsigned> (time(0)));
    initalizePhonemeSet();
    outputSize = phonemeSet.size();

    std::string response;
    SAMPLE_RATE = 16000;
    std::cout << "Select sample rate (default: " << SAMPLE_RATE << ")\n";
    std::getline(std::cin, response);
    if (response != "") {
        SAMPLE_RATE = std::stoi(response); // Exits if unparsable
        if (SAMPLE_RATE <= 0) {
            assert("Out of range");
        }
    }

    window = new float[FFT_FRAME_SAMPLES];
    for (int i = 0; i < FFT_FRAME_SAMPLES; i++) {
        //window[i] = 1.0; // None
        //window[i] = 0.5f * (1.0f - cos((6.2831853f * i) / FFT_FRAME_SAMPLES)); // Hann
        //window[i] = 0.5f * (1.0f - cos((6.2831853f * i) / FFT_FRAME_SAMPLES)) * pow(2.7182818f, (-5.0f * abs(FFT_FRAME_SAMPLES - 2.0f * i)) / FFT_FRAME_SAMPLES); // Hann - Poisson
        //window[i] = 0.355768f - 0.487396f * cosf((6.28318530f * i) / FFT_FRAME_SAMPLES) - 0.144232 * cosf((12.5663706f * i) / FFT_FRAME_SAMPLES) - 0.012604 * cosf((18.8495559f * i) / FFT_FRAME_SAMPLES); // Nuttall
        window[i] = 0.3635819 - 0.4891775 * cosf((6.28318530f * i) / FFT_FRAME_SAMPLES) - 0.1365995 * cosf((12.5663706f * i) / FFT_FRAME_SAMPLES) - 0.0106411 * cosf((18.8495559f * i) / FFT_FRAME_SAMPLES); // Blackman - Nuttall
    }

    fftwIn = (float*)fftw_malloc(sizeof(float) * FFT_FRAME_SAMPLES);
    fftwOut = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * FFT_REAL_SAMPLES);
    fftwPlan = fftwf_plan_dft_r2c_1d(FFT_FRAME_SAMPLES, fftwIn, fftwOut, FFTW_MEASURE);

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
    
    // Cleanup
    fftwf_destroy_plan(fftwPlan);
    free(fftwIn);
    fftwf_free(fftwOut);
    delete[] window;

    return error;
}