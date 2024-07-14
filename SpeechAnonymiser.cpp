#include "define.h"

#define DR_MP3_IMPLEMENTATION
#define DR_WAV_IMPLEMENTATION

#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <thread>
#include <filesystem>
#include <random>
#include <unordered_set>
#include <rtaudio/RtAudio.h>
#include <fftw3.h>
#include <cargs.h>
#include <samplerate.h>
#include <dr_mp3.h>
#include <dr_wav.h>
#include "Visualizer.h"
#include "TSVReader.h"
#include "NeuralNetwork.h"

float* window;
float* fftwIn;
fftwf_complex* fftwOut;
fftwf_plan fftwPlan;

std::unordered_set<size_t> phonemeSet = {

};

struct InputData {
    INPUT_TYPE** buffer;
    size_t bufferBytes;
    size_t totalFrames;
    size_t channels;
    size_t writeOffset;
    size_t lastProcessed;
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

    void loadMP3(int targetSampleRate) {
        // Read mp3 file from tsv
        std::cout << "Loading MP3: " << tsvElements[TSVReader::Indices::PATH] << '\n';

        drmp3_config cfg;
        drmp3_uint64 samples;
        float* floatBuffer = drmp3_open_file_and_read_pcm_frames_f32((clipPath + tsvElements[TSVReader::Indices::PATH]).c_str(), &cfg, &samples, NULL);

        if (cfg.sampleRate == targetSampleRate) {
            memcpy(buffer, floatBuffer, samples);
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
    }

    Clip() {
        clipPath = "";
        tsvElements = NULL;
        buffer = new float[48000 * CLIP_LENGTH];
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
    float* real;
    float* imaginary;
    double volume;
    double pitch;

    Frame() {
        real = new float[FFT_REAL_SAMPLES];
        imaginary = new float[FFT_REAL_SAMPLES];
    }
    Frame(size_t size) {
        real = new float[size];
        imaginary = new float[size];
    }
};

std::chrono::time_point start = std::chrono::system_clock::now();

int processInput(void* /*outputBuffer*/, void* inputBuffer, unsigned int nBufferFrames,
    double /*streamTime*/, RtAudioStreamStatus /*status*/, void* data) {

    InputData* iData = (InputData*)data;

    if ((iData->lastProcessed - iData->writeOffset) % iData->totalFrames < nBufferFrames) {
        auto sinceStart = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::system_clock::now() - start);
        std::cout << sinceStart.count() << ": Stream overflow detected!\n";
        iData->writeOffset = (iData->lastProcessed + nBufferFrames + 128) % iData->totalFrames;
    }

    for (size_t i = 0; i < nBufferFrames; i++) {
        size_t writePosition = (iData->writeOffset + i) % iData->totalFrames;
        for (int j = 0; j < iData->channels; j++) {
            iData->buffer[j][writePosition] = ((INPUT_TYPE*)inputBuffer)[i * 2 + j];
        }
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
    for (unsigned int i = 0; i < nBufferFrames; i++) {
        unsigned int readPosition = (startSample + i) % iData->totalFrames;
        for (unsigned int j = 0; j < oData->channels; j++) {
            INPUT_TYPE input = iData->buffer[j][readPosition];
            *buffer++ = (OUTPUT_TYPE)((input * OUTPUT_SCALE) / INPUT_SCALE);
        }
    }
    oData->lastSample = startSample + nBufferFrames;
    iData->lastProcessed = oData->lastSample;

    return 0;
}

void cleanupRtAudio(RtAudio audio) {
    if (audio.isStreamOpen()) {
        audio.closeStream();
    }
}

void processFrame(Frame& frame, const float* audio, const size_t& start, const size_t& totalSize) {
    for (int i = 0; i < FFT_FRAME_SAMPLES; i++) {
        unsigned int readLocation = (start + i) % totalSize;
        fftwIn[i] = audio[readLocation] * window[i];
    }
    fftwf_execute(fftwPlan);
    for (int i = 0; i < FFT_REAL_SAMPLES; i++) {
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
    for (int i = 0; i < FFT_FRAMES; i++) {
        app.fftData.frequencies[i] = new float[FFT_REAL_SAMPLES];
        for (int j = 0; j < FFT_REAL_SAMPLES; j++) {
            app.fftData.frequencies[i][j] = 0.0;
        }
    }
    std::thread fft = std::thread([&app, &inputData] {
        // Setup FFT

        // Wait for visualization to open
        while (!app.isOpen) {
            Sleep(10);
        }
        Frame frame = Frame(FFT_FRAME_SAMPLES);
        size_t lastSampleStart = 0;
        while (app.isOpen) {
            // Wait for enough samples to be recorded to pass to FFT
            while ((inputData.writeOffset - lastSampleStart) % inputData.totalFrames < FFT_FRAME_SAMPLES) {
                //auto start = std::chrono::high_resolution_clock::now();
                Sleep(10);
                //auto actual = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
                //std::cout << actual.count() << '\n';
            }
            lastSampleStart = (lastSampleStart + FFT_FRAME_SPACING) % inputData.totalFrames;
            // Do FFT stuff
            processFrame(frame, inputData.buffer[0], lastSampleStart, inputData.totalFrames);
            // Write FFT output to visualizer
            memcpy(app.fftData.frequencies[app.fftData.currentFrame], frame.real, sizeof(float) * FFT_REAL_SAMPLES);
            app.fftData.currentFrame = (app.fftData.currentFrame + 1) % FFT_FRAMES;
        }
        });

#ifdef HIDE_CONSOLE
    ShowWindow(GetConsoleWindow(), SW_HIDE);
#else
    ShowWindow(GetConsoleWindow(), SW_SHOW);
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
    InputData inputData = InputData();
    inputData.totalFrames = (unsigned long)(48000 * INPUT_BUFFER_TIME);
    inputData.channels = 1;
    size_t totalBytes;
    totalBytes = sizeof(INPUT_TYPE) * inputData.totalFrames * 1;
    inputData.buffer[1] = { new INPUT_TYPE[totalBytes] };
    for (unsigned int i = 0; i < inputData.totalFrames; i++) {
        inputData.buffer[i] = 0;
    }
    inputData.writeOffset = 0;

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

    std::vector<Phone> phones;
    std::cout << "Training mode\n";

    TSVReader tsv;
    tsv.open(combine(path, "/train.tsv"));
    const std::string clipPath = std::string(combine(path, "/clips/"));
    const std::string transcriptsPath = std::string(combine(path, "/transcript/"));

    unsigned long fftStart = 0;
    unsigned int currentFrame = 0;
    Frame* frames = new Frame[FFT_FRAMES];
    for (int i = 0; i < FFT_FRAMES; i++) {
        frames[i].real = new float[FFT_REAL_SAMPLES];
        float* values = new float[FFT_FRAME_SAMPLES];
        std::fill_n(values, FFT_FRAME_SAMPLES, 0.0f);
        processFrame(frames[i], values, 0, FFT_FRAME_SAMPLES);
    }

    size_t inputSize = FFT_REAL_SAMPLES;
    size_t outputSize = 44;
    std::vector<size_t> neurons = {
        inputSize,
        50,
        50,
        outputSize,
    };
    NeuralNetwork neural = NeuralNetwork(neurons);

    std::hash<std::string> hasher;
    bool activeClip1 = true; // Currently reading from clip 1
    Clip clip1, clip2;
    loadNextClip(clipPath, tsv, clip1, 48000);
    float* neuralInput = new float[inputSize];
    float* neuralOutput = new float[outputSize];
    while (tsv.good()) {
        Clip* currentClip;

        std::thread clipLoader;
        if (activeClip1) {
            clipLoader = std::thread(loadNextClip, std::ref(clipPath), std::ref(tsv), std::ref(clip2), 48000);
            currentClip = &clip1;
        } else {
            clipLoader = std::thread(loadNextClip, std::ref(clipPath), std::ref(tsv), std::ref(clip1), 48000);
            currentClip = &clip2;
        }
        activeClip1 = !activeClip1;

#pragma region Really shitty textgrid parser
        // Get transcription
        const std::string path = currentClip->tsvElements[TSVReader::Indices::PATH];
        const std::string transcriptionPath = transcriptsPath + currentClip->tsvElements[TSVReader::Indices::CLIENT_ID] + "/" + path.substr(0, path.length() - 4) + ".TextGrid";
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
            p.minIdx = 48000 * p.min;
            p.maxIdx = 48000 * p.max;
            p.phonetic = hasher(text);
        }

        reader.close();
#pragma endregion

        fftStart = std::rand() % FFT_FRAME_SPACING;
        currentFrame = 0;
        for (int i = 0; i < FFT_FRAMES; i++) {
            float* values = new float[FFT_FRAME_SAMPLES];
            std::fill_n(values, FFT_FRAME_SAMPLES, 0.0f);
            processFrame(frames[i], values, 0, FFT_FRAME_SAMPLES);
            delete(values);
        }
        while ((size_t)fftStart + FFT_FRAME_SAMPLES < currentClip->size) {
            processFrame(frames[currentFrame], currentClip->buffer, fftStart, currentClip->size);

            // The neural network thing
            // Pass all frames to model, try to identify phoneme a few frames back
            // ex: 7 total frames, 2 frames latency
            // 1, 2, 3, 4, 5(identify phoneme located here), 6, 7(current processed)
            // Might be more accurate if it has some information before and after
            //for (int i = 0; i < FFT_FRAMES; i++) {
            //    int readFrame = (currentFrame + 1 + i) % FFT_FRAMES;
            //    memcpy(neuralInput + FFT_FRAME_SAMPLES * i, frames[readFrame].values, FFT_FRAME_SAMPLES);
            //}
            memcpy(neuralInput, frames[currentFrame].real, FFT_REAL_SAMPLES);
            neural.process(neuralInput, inputSize, neuralOutput, outputSize);

            currentFrame = (currentFrame + 1) % FFT_FRAMES;
            fftStart += FFT_FRAME_SPACING;
        }

        clipLoader.join();
    }
    return 0;
}

int commandPreprocess(const char* path) {
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
            std::cout << "Iteration took " << std::chrono::duration_cast<std::chrono::duration<double>>(diff) << " seconds\n";
        }
    }

    return 0;
}

int commandDefault() {

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
    inputParameters.nChannels = inputInfo.inputChannels;
    unsigned int sampleRate = inputInfo.preferredSampleRate;
    unsigned int bufferFrames = INPUT_BUFFER_SIZE;

    InputData inputData = InputData();
    inputData.totalFrames = (unsigned long)(sampleRate * INPUT_BUFFER_TIME);
    inputData.channels = inputParameters.nChannels;
    size_t totalBytes = sizeof(INPUT_TYPE) * inputData.totalFrames;
    inputData.buffer = new INPUT_TYPE*[inputParameters.nChannels];
    for (int i = 0; i < inputParameters.nChannels; i++) {
        inputData.buffer[i] = new INPUT_TYPE[totalBytes];
        for (unsigned int j = 0; j < inputData.totalFrames; j++) {
            inputData.buffer[i][j] = 0;
        }
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

    return 0;
}

int main(int argc, char** argv) {
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
