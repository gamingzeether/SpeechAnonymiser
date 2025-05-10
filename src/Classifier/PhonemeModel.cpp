#include "PhonemeModel.hpp"

#include <filesystem>
#include <fstream>

#define ARCHIVE_FILE "classifier.zip"
#define MODEL_FILE "phoneme_model.bin"
#define CONFIG_FILE "classifier.json"
#define DEFAULT_CONFIG_FILE "default_classifier.json"
#define ZIP_FILES { MODEL_FILE, CONFIG_FILE }

#define _VC mlpack::NaiveConvolution<mlpack::ValidConvolution>
#define _FC mlpack::NaiveConvolution<mlpack::FullConvolution>

// Define Layer types
#define DROPOUT         mlpack::DropoutType<MAT_TYPE>

#define TANH_ACTIVATION mlpack::TanHType<MAT_TYPE>
#define RELU_ACTIVATION mlpack::LeakyReLUType<MAT_TYPE>
#define LOG_SOFTMAX     mlpack::LogSoftMaxType<MAT_TYPE>

#define LINEAR          mlpack::LinearType<MAT_TYPE>
#define LINEARNB        mlpack::LinearNoBiasType<MAT_TYPE>
#define LSTM            mlpack::LSTMType<MAT_TYPE>
#define CONVOLUTION     mlpack::ConvolutionType<_VC, _FC, _VC, MAT_TYPE>
#define POOLING         mlpack::MaxPoolingType<MAT_TYPE>

void PhonemeModel::setHyperparameters(Hyperparameters hp) {
  this->hp = hp;
}

void PhonemeModel::initModel() {
  net = NETWORK_TYPE();

  // Default architecture defined in PhonemeModel::setDefaultModel()
  JSONHelper::JSONObj layers = config.object()["layers"];
  size_t numLayers = layers.get_array_size();
  for (size_t i = 0; i < numLayers; i++)
    addFromJson(layers[i]);
}

void PhonemeModel::initOptimizer() {
  optim.BatchSize() = hp.batchSize();
  optim.StepSize() = hp.stepSize();
  optim.Shuffle() = false;
}

void PhonemeModel::save(int checkpoint) {
  auto tempPath = getTempPath();
  ModelSerializer::saveNetwork(tempPath + MODEL_FILE, &net);
  config.save();

  int error = 0;
  zip_t* archive = zip_open(ARCHIVE_FILE, ZIP_CHECKCONS | ZIP_CREATE, &error);
  if (archive == NULL) {
    logZipError(error);
    return;
  }
  for (std::string string : ZIP_FILES) {
    zip_source_t* source = zip_source_file(archive, (tempPath + string).c_str(), 0, ZIP_LENGTH_TO_END);
    zip_int64_t index = zip_file_add(archive, string.c_str(), source, ZIP_FL_OVERWRITE | ZIP_FL_ENC_UTF_8);
    if (index < 0)
      logZipError(archive);
  }
  error = zip_close(archive);
  if (error != 0)
    logZipError(archive);
  cleanUnpacked();

  if (checkpoint >= 0) {
    std::filesystem::copy_file(
      ARCHIVE_FILE, 
      ARCHIVE_FILE + std::to_string(checkpoint),
      std::filesystem::copy_options::overwrite_existing);
  }
}

bool PhonemeModel::load() {
  auto tempPath = getTempPath();
  // Try open classifier.zip
  if (std::filesystem::exists(ARCHIVE_FILE)) {
    int error = 0;
    zip_t* archive = zip_open(ARCHIVE_FILE, ZIP_CHECKCONS, &error);
    if (archive == NULL) {
      logZipError(error);
      return false;
    }
    size_t bufferSize = 16 * 1024 * 1024;
    char* buf = new char[bufferSize];
    for (std::string string : ZIP_FILES) {
      zip_file_t* file = zip_fopen(archive, string.c_str(), ZIP_FL_UNCHANGED);
      if (file == NULL)
        logZipError(archive);
      std::ofstream out = std::ofstream(tempPath + string, std::ios::binary | std::ios::trunc);
      while (true) {
        zip_int64_t readBytes = zip_fread(file, buf, bufferSize);
        if (readBytes <= 0)
          break;
        out.write(buf, readBytes);
      };
      out.close();
      zip_fclose(file);
      if (error != 0)
        logZipError(archive);
    }
    delete[] buf;
    zip_discard(archive);
  } else {
    G_LG(Util::format("%s does not exist", ARCHIVE_FILE), Logger::DBUG);
  }

  // This should only be true if doing something with the dataset (training or evaluating)
  bool hasPhonemeSet = Global::get().isClassifierSet();
  std::string psName;
  if (hasPhonemeSet) {
    psName = Global::get().getClassifierPhonemeSetName();
  }

  // Load config file
  config = Config(tempPath + CONFIG_FILE, 0);
  bool configLoaded = config.loadDefault(DEFAULT_CONFIG_FILE);
  auto& defObj = config.defaultObject();
  bool resetConfig = defObj["input_features"].get_int() != inputSize ||
      (hasPhonemeSet && defObj["phoneme_set"].get_string() != psName) ||
      defObj["sample_rate"].get_int() != sampleRate;
  if (configLoaded && !resetConfig) {
    G_LG(Util::format("Using classifier config from file '%s'", DEFAULT_CONFIG_FILE), Logger::INFO);
  } else {
    G_LG(Util::format("Generating new classifier config '%s'", DEFAULT_CONFIG_FILE), Logger::WARN);
    config.setDefault("input_features", inputSize)
        .setDefault("phoneme_set", psName)
        .setDefault("sample_rate", sampleRate);
    outputSize = G_PS_C.size();
    setDefaultModel();
    config.saveDefault(DEFAULT_CONFIG_FILE);
  }
  config.load();

  // Load or initalize model network and optimizer
  bool loaded = true;
  if (!config.matchesDefault()) {
    loaded = false;
    config.useDefault();
    G_LG("Config does not match", Logger::DBUG);
  }
  if (loaded && !ModelSerializer::loadNetwork(tempPath + MODEL_FILE, &net)) {
    loaded = false;
    G_LG(Util::format("Failed to load model %s", (tempPath + MODEL_FILE).c_str()), Logger::DBUG);
  }
  cleanUnpacked();

  if (hasPhonemeSet) {
    // Save in case it is empty
    config.object()["phoneme_set"] = Global::get().getClassifierPhonemeSetName();
  } else {
    // Load from config
    Global::get().setClassifierPhonemeSet(defObj["phoneme_set"].get_string());
  }
  outputSize = G_PS_C.size();

  if (!loaded)
    initModel();
  net.InputDimensions() = { FFT_FRAMES, FRAME_SIZE, 3 };
  net.BPTTSteps() = hp.bpttSteps();
  net.MemoryInit();
  initOptimizer();

  printInfo();

  return loaded;
}

void PhonemeModel::cleanUnpacked() {
  auto tempPath = getTempPath();
  for (const char* string : ZIP_FILES) {
    std::filesystem::remove(tempPath + string);
  }
}

void PhonemeModel::logZipError(int error) {
  zip_error_t zerror;
  zip_error_init_with_code(&zerror, error);
  logZipError(&zerror);
}

void PhonemeModel::logZipError(zip_t* archive) {
  zip_error_t* error = zip_get_error(archive);
  logZipError(error);
}

void PhonemeModel::logZipError(zip_error_t* error) {
  const char* errString = zip_error_strerror(error);
  G_LG(errString, Logger::ERRO);
  zip_error_fini(error);
}

// Utility macros
#define _ADD_LAYER_PARAMS_0()
#define _ADD_LAYER_PARAMS_1(Name, Val     ) layer[ QUOTE(Name) ] = Val;
#define _ADD_LAYER_PARAMS_2(Name, Val, ...) layer[ QUOTE(Name) ] = Val; _ADD_LAYER_PARAMS_1(__VA_ARGS__)
#define _ADD_LAYER_PARAMS_3(Name, Val, ...) layer[ QUOTE(Name) ] = Val; _ADD_LAYER_PARAMS_2(__VA_ARGS__)
#define _ADD_LAYER_PARAMS_4(Name, Val, ...) layer[ QUOTE(Name) ] = Val; _ADD_LAYER_PARAMS_3(__VA_ARGS__)
#define _ADD_LAYER_PARAMS_5(Name, Val, ...) layer[ QUOTE(Name) ] = Val; _ADD_LAYER_PARAMS_4(__VA_ARGS__)
#define ADD_LAYER(Type, NArgs, ...) \
{ \
  JSONHelper::JSONObj layer = layers.append(); \
  layer["type"] = std::string(STRINGIFY(Type)); \
  _ADD_LAYER_PARAMS_##NArgs(__VA_ARGS__); \
}

void PhonemeModel::setDefaultModel() {
  JSONHelper::JSONObj configRoot = config.defaultObject().getRoot();
  JSONHelper::JSONObj layers;
  if (configRoot.exists("layers")) {
    layers = configRoot["layers"];
    layers.clear_arr();
  } else {
    layers = configRoot.add_arr("layers");
  }
  
  // Convolutional block
  for (size_t i = 0; i < 3; i++) {
    ADD_LAYER(dropout, 1,
        ratio, 0.1);
    ADD_LAYER(conv2d, 5,
        maps, 16,
        height, 2,
        width, 2,
        stride_x, 2,
        stride_y, 2);
    ADD_LAYER(tanh, 0);
  }

  // Dense layers
  for (size_t i = 0; i < 5; i++) {
    ADD_LAYER(dropout, 1,
        ratio, 0.1);
    ADD_LAYER(linear, 1,
        out_size, 256);
    ADD_LAYER(relu, 0);
  }

  // LSTM
  ADD_LAYER(dropout, 1,
      ratio, 0.1);
  ADD_LAYER(lstm, 1,
      out_size, 256);
  ADD_LAYER(relu, 0);

  // More dense layers
  for (size_t i = 0; i < 2; i++) {
    ADD_LAYER(dropout, 1,
        ratio, 0.1);
    ADD_LAYER(linear, 1,
        out_size, 256);
    ADD_LAYER(relu, 0);
  }

  // Output layer
  ADD_LAYER(dropout, 1,
      ratio, 0.1);
  ADD_LAYER(linear, 1,
      out_size, outputSize);
  ADD_LAYER(log_softmax, 0);
}
#undef _ADD_LAYER_PARAMS_1
#undef _ADD_LAYER_PARAMS_2
#undef _ADD_LAYER_PARAMS_3
#undef _ADD_LAYER_PARAMS_4
#undef ADD_LAYER

void PhonemeModel::addFromJson(const JSONHelper::JSONObj& layer) {
  // Add the layer
  std::string type = layer["type"].get_string();
  // ========================== Dropout ==========================
  if (type == "dropout") {
    double ratio = layer["ratio"].get_real();
    net.Add<DROPOUT>(ratio);
  // ==================== Activation Functions ====================
  } else if (type == "tanh") {
    net.Add<TANH_ACTIVATION>();
  } else if (type == "relu") {
    net.Add<RELU_ACTIVATION>();
  } else if (type == "log_softmax") {
    net.Add<LOG_SOFTMAX>();
  // ====================== Connected layers ======================
  } else if (type == "linear") {
    int outSize = layer["out_size"].get_int();
    net.Add<LINEAR>(outSize);
  } else if (type == "linear_nb") {
    int outSize = layer["out_size"].get_int();
    net.Add<LINEARNB>(outSize);
  } else if (type == "lstm") {
    int outSize = layer["out_size"].get_int();
    net.Add<LSTM>(outSize);
  } else if (type == "conv2d") {
    int maps = layer["maps"].get_int();
    int width = layer["width"].get_int();
    int height = layer["height"].get_int();
    int strideX = layer["stride_x"].get_int();
    int strideY = layer["stride_y"].get_int();
    net.Add<CONVOLUTION>(maps, width, height, strideX, strideY);
  } else if (type == "pool2d") {
    int width = layer["width"].get_int();
    int height = layer["height"].get_int();
    int strideX = layer["stride_x"].get_int();
    int strideY = layer["stride_y"].get_int();
    net.Add<POOLING>(width, height, strideX, strideY);
  }
}

std::string PhonemeModel::getTempPath() {
  std::string tempPath = std::filesystem::temp_directory_path().string();
  // Ensure tempPath has a trailing slash
  if (tempPath.back() != '\\' && tempPath.back() != '/')
    tempPath += '/';
  tempPath += "SpeechAnonymiser/";
  if (!std::filesystem::exists(tempPath))
    std::filesystem::create_directories(tempPath);
  return tempPath;
}

void PhonemeModel::printInfo() {
  NETWORK_TYPE copy = net;
  // Forward pass to initalize network
  CUBE_TYPE in(inputSize, 1, 1);
  CUBE_TYPE out;
  copy.Predict(in, out);
  // Header
  {
    std::string line = "";
    line += Util::leftPad("Layer Type", 20);
    line += Util::leftPad("Input Dimensions", 20);
    line += Util::leftPad("Output Dimensions", 20);
    line += Util::leftPad("Parameters", 20);
    G_LG(line, Logger::INFO);
  }
  for (auto layer : copy.Network()) {
    printLayer(layer);
  }
}

#define _TRY(Type) if (_printLayer(dynamic_cast<Type*>(layer), #Type)) return;
void PhonemeModel::printLayer(mlpack::Layer<MAT_TYPE>* layer) {
  _TRY(DROPOUT        );
  _TRY(TANH_ACTIVATION);
  _TRY(RELU_ACTIVATION);
  _TRY(LOG_SOFTMAX    );
  _TRY(LINEAR         );
  _TRY(LINEARNB       );
  _TRY(LSTM           );
  _TRY(CONVOLUTION    );
  _TRY(POOLING        );
}
#undef _TRY
