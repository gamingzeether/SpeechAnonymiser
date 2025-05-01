#include "define.hpp"

#include "common_inc.hpp"

#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <thread>
#include <filesystem>
#include <random>
#include <unordered_set>
#include <optional>
#include <cargs.h>
#include "Classifier/PhonemeClassifier.hpp"
#include "Classifier/Train/Dataset.hpp"
#include "Classifier/Train/CVIterator.hpp"
#include "Utils/ClassifierHelper.hpp"
#include "Utils/Global.hpp"
#include "Utils/TranslationMap.hpp"
#include "SpeechEngine/SpeechEngineConcatenator.hpp"
#include "SpeechEngine/SpeechEngineFormant.hpp"
#include "structs.hpp"
#ifdef AUDIO
#include <rtaudio/RtAudio.h>
#endif
#ifdef GUI
#include "GUI/Visualizer.hpp"
#include "Debugging/InteractiveDebugger.hpp"
#endif

#define ERROR_STUB(Requires, Method, ...) Method(__VA_ARGS__) { G_LG(STRINGIFY(Method) " requires compiling with " STRINGIFY(Requires) " support", Logger::DEAD); }

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
  .description = "Evaluate model accuracy"},

  {
   .identifier = '1',
   .access_name = "tune",
   .value_name = "[Dataset directory]",
   .description = "Tune training hyperparameters"}
};

template <typename T>
bool requestString(const std::string& request, std::string& out, T& value) {
  std::cout << request << std::endl << "(Default " << value << ") ";
  bool isInteractive = getenv("NONINTERACTIVE") == NULL;
  if (isInteractive) {
    std::getline(std::cin, out);
  }
  G_LG(Util::format("Requested: '%s', Default: '%s', Recieved: '%s'",
      request.c_str(), std::to_string(value).c_str(), out.c_str()), Logger::DBUG);
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
      G_LG("Stream overflow detected", Logger::ERRO);
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
    G_LG("Stream underflow detected", Logger::ERRO);
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
        input = Util::lerp(p1, p2, realRead - floor);
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
  std::unique_lock lock(container.mtx);
  const size_t channels = 2;
  if (container.pause) {
    std::fill_n(buffer, nBufferFrames * channels, 0);
  } else {
    size_t size = container.audio.size();
    size_t& ptr = container.pointer;
    for (size_t i = 0; i < nBufferFrames; i++) {
      float data = (ptr < size) ? container.audio[ptr++] : 0;
      for (size_t j = 0; j < channels; j++) {
        *buffer++ = data;
      }
    }
  }
  return 0;
}

void cleanupRtAudio(RtAudio audio) {
  if (audio.isStreamOpen()) {
    audio.closeStream();
  }
}

// Requests user to select an audio device for either input or output
// Returns true on success, false on error
bool selectAudioDevice(const bool input, OUT AudioDevice& audioDev) {
  // Check if there are any devices available
  int defaultDevice = (input) ? audioDev.audio.getDefaultInputDevice() : audioDev.audio.getDefaultOutputDevice();
  const char* deviceType = (input) ? "input" : "output";
  if (defaultDevice == 0) {
    G_LG(Util::format("No %s device available", deviceType), Logger::ERRO);
    return false;
  }

  // Select target devices
  std::vector<RtAudio::DeviceInfo> audioDevices = std::vector<RtAudio::DeviceInfo>();
  auto devices = audioDev.audio.getDeviceIds();
  for (size_t i = 0; i < devices.size(); i++) {
    RtAudio::DeviceInfo deviceInfo = audioDev.audio.getDeviceInfo(devices[i]);
    bool isTargetType = (input && deviceInfo.inputChannels > 0) ||
        (!input && deviceInfo.outputChannels > 0);
    bool isDefault = (input && deviceInfo.isDefaultInput) ||
        (!input && deviceInfo.isDefaultOutput);

    if (isDefault)
      defaultDevice = audioDevices.size(); // Get the index that this device will be at
    if (isTargetType)
      audioDevices.push_back(std::move(deviceInfo));
  }

  // Request user to select an audio device
  for (size_t i = 0; i < audioDevices.size(); i++) {
    std::cout << i << ": " << audioDevices[i].name << std::endl;
  }
  requestInput(Util::format("Select %s device", deviceType), defaultDevice);
  if (defaultDevice < 0 || defaultDevice >= audioDevices.size()) {
    G_LG("Selected device is out of range", Logger::ERRO);
    return false;
  } else {
    defaultDevice = audioDevices[defaultDevice].ID;
  }

  // Set up stream parameters
  RtAudio::DeviceInfo deviceInfo = audioDev.audio.getDeviceInfo(defaultDevice);
  audioDev.streamParameters = RtAudio::StreamParameters();
  audioDev.streamParameters.deviceId = defaultDevice;
  audioDev.streamParameters.nChannels = (input) ? deviceInfo.inputChannels : deviceInfo.outputChannels;

  // Get device properties
  audioDev.samplerate = deviceInfo.preferredSampleRate;
  return true;
}
#else
ERROR_STUB(audio, int processInput, void* /*outputBuffer*/, void* inputBuffer, unsigned int nBufferFrames,
  double /*streamTime*/, RtAudioStreamStatus /*status*/, void* data)
ERROR_STUB(audio, int processOutput, void* outputBuffer, void* /*inputBuffer*/, unsigned int nBufferFrames,
  double /*streamTime*/, RtAudioStreamStatus status, void* data)
ERROR_STUB(audio, int oneshotOutput, void* outputBuffer, void* /*inputBuffer*/, unsigned int nBufferFrames,
  double /*streamTime*/, RtAudioStreamStatus status, void* data)
ERROR_STUB(audio, void cleanupRtAudio, RtAudio audio)
ERROR_STUB(audio, void selectAudioDevice, bool input)
ERROR_STUB(audio, void startFFT, InputData& inputData)
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
    CPU_CUBE_TYPE data, labels;
    Dataset::makeCubes(data, labels, 1, 1);

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
      std::array<float, FRAME_SIZE> frameData;
      for (size_t i = 0; i < FRAME_SIZE; i++)
        frameData[i] = frame.avg[i];
      app.updateWaterfall(frameData);

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

void startDebugger(size_t sampleRate, std::string dataPath, AudioContainer& ac) {
  InteractiveDebugger debugger(sampleRate, dataPath, ac);
  debugger.run();
}
#else
ERROR_STUB(GUI, void startFFT, InputData& inputData)
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
  G_LG(Util::format("Set examples count: %d", examples), Logger::DBUG);
  if (examples <= 0) {
    throw("Out of range");
  }

  int epochs = 1000;
  requestInput("Set number of epochs", epochs);
  G_LG(Util::format("Set epochs: %d", epochs), Logger::DBUG);
  if (epochs <= 0) {
    throw("Out of range");
  }

  //double stepSize = STEP_SIZE;
  //requestInput("Set training rate", stepSize);
  //G_LG(Util::format("Set training rate: %lf", stepSize), Logger::DBUG);
  //if (stepSize <= 0) {
  //  throw("Out of range");
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

int commandDefault() {
  // Set up audio devices
  AudioDevice inputDevice, outputDevice;
  inputDevice.bufferFrames = INPUT_BUFFER_SIZE;
  outputDevice.bufferFrames = OUTPUT_BUFFER_SIZE;
  if (!selectAudioDevice(true, inputDevice))
    G_LG("Failed to get audio input", Logger::DEAD);
  if (!selectAudioDevice(false, outputDevice))
    G_LG("Failed to get audio output", Logger::DEAD);

  InputData inputData = InputData(inputDevice.streamParameters.nChannels, sampleRate * INPUT_BUFFER_TIME);

  OutputData outputData = OutputData();
  outputData.lastValues = (double*)calloc(outputDevice.streamParameters.nChannels, sizeof(double));
  outputData.channels = outputDevice.streamParameters.nChannels;
  outputData.input = &inputData;
  outputData.scale = (double)outputDevice.samplerate / sampleRate;

  // Open audio streams
  if (inputDevice.audio.openStream(NULL, &inputDevice.streamParameters, INPUT_FORMAT,
    sampleRate, &inputDevice.bufferFrames, &processInput, (void*)&inputData, &inputDevice.flags)) {
    G_LG(inputDevice.audio.getErrorText(), Logger::ERRO);
    return 0; // problem with device settings
  }

  if (outputDevice.audio.openStream(&outputDevice.streamParameters, NULL, OUTPUT_FORMAT,
    outputDevice.samplerate, &outputDevice.bufferFrames, &processOutput, (void*)&outputData, &outputDevice.flags)) {
    G_LG(outputDevice.audio.getErrorText(), Logger::ERRO);
    return 0; // problem with device settings
  }

  // Start audio streams
  if (inputDevice.audio.startStream()) {
    G_LG(inputDevice.audio.getErrorText(), Logger::ERRO);
    cleanupRtAudio(inputDevice.audio);
    return -1;
  }

  if (outputDevice.audio.startStream()) {
    G_LG(outputDevice.audio.getErrorText(), Logger::ERRO);
    cleanupRtAudio(outputDevice.audio);
    return -1;
  }

  // Setup data visualization
  startFFT(inputData);

  return 0;
}

// For development - listening to and comparing sound clips/phonemes
int commandInteractive(const std::string& path) {
  AudioContainer ac;
  ac.audio = std::vector<float>(1);
  ac.pointer = 0;

  // Set up audio devices
  AudioDevice outputDevice;
  outputDevice.bufferFrames = OUTPUT_BUFFER_SIZE;
  if (!selectAudioDevice(false, outputDevice))
    G_LG("Failed to get audio output", Logger::DEAD);
  
  // Open audio streams
  outputDevice.samplerate = 16000;
  if (outputDevice.audio.openStream(&outputDevice.streamParameters, NULL, OUTPUT_FORMAT,
    outputDevice.samplerate, &outputDevice.bufferFrames, &oneshotOutput, (void*)&ac, &outputDevice.flags)) {
    G_LG(outputDevice.audio.getErrorText(), Logger::ERRO);
    return 0; // problem with device settings
  }

  // Start audio streams
  if (outputDevice.audio.startStream()) {
    G_LG(outputDevice.audio.getErrorText(), Logger::ERRO);
    cleanupRtAudio(outputDevice.audio);
    return -1;
  }

  startDebugger(outputDevice.samplerate, path, ac);
  
  return 0;
}

int commandEvaluate(const std::string& path) {
  classifier.evaluate(path);
  return 0;
}

int commandTune(const std::string& path) {
  int iterations = 1000;
  requestInput("Number of iterations", iterations);
  int radius = 2;
  requestInput("Steps in each direction", radius);
  classifier.tuneHyperparam(path, iterations, radius);
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
  G_LG(Util::format("Set sample rate: %d", sampleRate), Logger::DBUG);

  {
    double forwardSamples = FFT_FRAME_SAMPLES + (CONTEXT_FORWARD * FFT_FRAME_SPACING);
    double backwardSamples = FFT_FRAME_SAMPLES + (CONTEXT_BACKWARD * FFT_FRAME_SPACING);
    double forwardMsec = 1000.0 * (forwardSamples / sampleRate);
    double backwardMsec = 1000.0 * (backwardSamples / sampleRate);
    G_LG(Util::format("Forward context: %dms; Backward context: %dms", (int)forwardMsec, (int)backwardMsec), Logger::INFO);
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
  G_LG(Util::format("Launch args: %s", launchString.c_str()), Logger::INFO);
  G_LG(Util::format("Working directory: %s", std::filesystem::current_path().c_str()), Logger::INFO);

  tryMakeDir("logs");
  tryMakeDir("configs/articulators");
  tryMakeDir("configs/animations/phonemes");
  
  Global::get().setSpeechEnginePhonemeSet("IPA");

  cag_option_context context;
  cag_option_init(&context, options, CAG_ARRAY_SIZE(options), argc, argv);
  bool trainMode = false, preprocessMode = false, helpMode = false, interactiveMode = false, evaluateMode = false, tuneMode = false;
  std::string tVal, pVal, wVal, dVal, aVal, oVal, iiVal, eVal, tuVal;
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
    case '1':
      tuneMode = true;
      tuVal = cag_option_get_value(&context);
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
    } else if (tuneMode) {
      Util::removeTrailingSlash(tuVal);
      error = commandTune(tuVal);
    } else if (interactiveMode) {
      Util::removeTrailingSlash(iiVal);
      error = commandInteractive(iiVal);
    } else if (evaluateMode) {
      Util::removeTrailingSlash(eVal);
      error = commandEvaluate(eVal);
    } else {
      error = commandDefault();
    }
  }

  classifier.destroy();

  G_LG(Util::format("Program exited with code %d", error), Logger::INFO);

  return error;
}
