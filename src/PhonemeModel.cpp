#include "PhonemeModel.hpp"

#include <filesystem>
#include <fstream>
#include <format>
#include "Global.hpp"

#define ARCHIVE_FILE "classifier.zip"
#define MODEL_FILE "phoneme_model.bin"
#define CONFIG_FILE "classifier.json"
#define DEFAULT_CONFIG_FILE "default_classifier.json"
#define ZIP_FILES { MODEL_FILE, CONFIG_FILE }

#define _VC mlpack::NaiveConvolution<mlpack::ValidConvolution>
#define _FC mlpack::NaiveConvolution<mlpack::FullConvolution>
#define CONVT _VC, _FC, _VC, MAT_TYPE

#define LINEARNB(neurons) net.Add<mlpack::LinearNoBiasType<MAT_TYPE, mlpack::L2Regularizer>>(neurons, mlpack::L2Regularizer(hp.l2()))
#define LINEAR(neurons) net.Add<mlpack::LinearType<MAT_TYPE, mlpack::L2Regularizer>>(neurons, mlpack::L2Regularizer(hp.l2()))
#define CONVOLUTION(maps, width, height, strideX, strideY) net.Add<mlpack::ConvolutionType<CONVT>>(maps, width, height, strideX, strideY);
#define TANH_ACTIVATION net.Add<mlpack::TanHType<MAT_TYPE>>()
#define RELU_ACTIVATION net.Add<mlpack::PReLUType<MAT_TYPE>>()
#define DROPOUT net.Add<mlpack::DropoutType<MAT_TYPE>>(hp.dropout())

void PhonemeModel::setHyperparameters(Hyperparameters hp) {
    this->hp = hp;
}

void PhonemeModel::initModel() {
    net = NETWORK_TYPE();
    
    int x = FFT_FRAMES, y = FRAME_SIZE, z = 3;

    // Default architecture defined in PhonemeModel::setDefaultModel()
    JSONHelper::JSONObj layers = config.object()["layers"];
    size_t numLayers = layers.get_array_size();
    for (size_t i = 0; i < numLayers; i++) {
        JSONHelper::JSONObj layer = layers[i];

        // Add dropout before a layer
        DROPOUT;

        // Add the layer
        std::string type = layer["type"].get_string();
        if (type == "conv2d") {
            int maps = layer["maps"].get_int();
            int width = layer["width"].get_int();
            int height = layer["height"].get_int();
            int strideX = layer["stride_x"].get_int();
            int strideY = layer["stride_y"].get_int();

            CONVOLUTION(maps, width, height, strideX, strideY);

            x = (x - width) / strideX;
            y = (y - height) / strideY;
            z = maps;
        } else if (type == "linear") {
            int neurons = layer["neurons"].get_int();

            LINEAR(neurons);
            x = neurons;
            y = 1;
            z = 1;
        }
        if (logger.has_value())
            logger->log(std::format("Layer {} output dimensions: \t{}\t{}\t{}\t({} features)", i, x, y, z, x * y * z), Logger::INFO);

        // Add activation function
        RELU_ACTIVATION;
    }

    // Add final output layers
    DROPOUT;
    LINEARNB(outputSize);
    net.Add<mlpack::LogSoftMaxType<MAT_TYPE>>();
}

void PhonemeModel::initOptimizer() {
    optim.BatchSize() = hp.batchSize();
    optim.StepSize() = hp.stepSize();
    optim.Shuffle() = true;
}

void PhonemeModel::save(int checkpoint) {
    ModelSerializer::saveNetwork(MODEL_FILE, &net);
    config.save();

    int error = 0;
    zip_t* archive = zip_open(ARCHIVE_FILE, ZIP_CHECKCONS | ZIP_CREATE, &error);
    if (archive == NULL) {
        logZipError(error);
        return;
    }
    for (const char* string : ZIP_FILES) {
        zip_source_t* source = zip_source_file(archive, string, 0, ZIP_LENGTH_TO_END);
        zip_int64_t index = zip_file_add(archive, string, source, ZIP_FL_OVERWRITE | ZIP_FL_ENC_UTF_8);
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
    bool loaded = true;
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
        for (const char* string : ZIP_FILES) {
            zip_file_t* file = zip_fopen(archive, string, ZIP_FL_UNCHANGED);
            if (file == NULL)
                logZipError(archive);
            std::ofstream out = std::ofstream(string, std::ios::binary | std::ios::trunc);
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
        loaded = true;
    } else if (logger.has_value()) {
        logger->log(std::format("{} does not exist", ARCHIVE_FILE), Logger::WARNING);
    }

    outputSize = G_PS.size();

    // Load config file
    config = Config(CONFIG_FILE, 0);
    bool configLoaded = config.loadDefault(DEFAULT_CONFIG_FILE);
    auto& defObj = config.defaultObject();
    bool configMatches = defObj["input_features"].get_int() == inputSize &&
        defObj["output_features"].get_int() == outputSize &&
        defObj["sample_rate"].get_int() == sampleRate;
    if (configLoaded && configMatches) {
        if (logger.has_value()) {
            logger->log(std::format("Using classifier config from file '{}'", DEFAULT_CONFIG_FILE), Logger::INFO);
        }
    } else {
        if (logger.has_value()) {
            logger->log(std::format("Generating new classifier config ", DEFAULT_CONFIG_FILE), Logger::INFO);
        }
        config.setDefault("input_features", inputSize)
            .setDefault("output_features", outputSize)
            .setDefault("sample_rate", sampleRate);
        setDefaultModel();
        config.saveDefault(DEFAULT_CONFIG_FILE);
    }
    config.load();

    // Load or initalize model network and optimizer
    if (!config.matchesDefault()) {
        loaded = false;
        config.useDefault();
        if (logger.has_value()) {
            logger->log("Config does not match", Logger::WARNING);
        }
    }
    if (loaded && !ModelSerializer::loadNetwork(MODEL_FILE, &net)) {
        loaded = false;
        if (logger.has_value()) {
            logger->log(std::format("Failed to load model {}", MODEL_FILE), Logger::WARNING);
        }
    }
    cleanUnpacked();
    if (!loaded)
        initModel();
    net.InputDimensions() = { FFT_FRAMES, FRAME_SIZE, 3 };
    initOptimizer();

    return loaded;
}

void PhonemeModel::cleanUnpacked() {
    for (const char* string : ZIP_FILES) {
        std::filesystem::remove(string);
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
    if (logger.has_value()) {
        logger.value().log(errString, Logger::ERR);
    }
    zip_error_fini(error);
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
    
    addConv(layers, 32, 2, 2, 1, 1);
    addConv(layers, 32, 2, 2, 1, 1);
    addConv(layers, 32, 2, 2, 1, 1);
    addConv(layers, 32, 2, 2, 1, 1);
    addLinear(layers, 256);
    addLinear(layers, 128);
}

void PhonemeModel::addConv(JSONHelper::JSONObj& layers, int maps, int width, int height, int strideX, int strideY) {
    JSONHelper::JSONObj layer = layers.append();
    layer["type"] = std::string("conv2d");

    layer["maps"] = maps;
    layer["width"] = width;
    layer["height"] = height;
    layer["stride_x"] = strideX;
    layer["stride_y"] = strideY;
}

void PhonemeModel::addLinear(JSONHelper::JSONObj& layers, int neurons) {
    JSONHelper::JSONObj layer = layers.append();
    layer["type"] = std::string("linear");

    layer["neurons"] = neurons;
}
