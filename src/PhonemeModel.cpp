#include "PhonemeModel.h"

#include <filesystem>
#include <fstream>
#include "ModelSerializer.h"
#include "Global.h"

#define ARCHIVE_FILE "classifier.zip"
#define MODEL_FILE "phoneme_model.bin"
#define CONFIG_FILE "classifier.json"
#define ZIP_FILES { MODEL_FILE, CONFIG_FILE }

#define CURRENT_VERSION -13

#define _VC mlpack::NaiveConvolution<mlpack::ValidConvolution>
#define _FC mlpack::NaiveConvolution<mlpack::FullConvolution>
#define CONVT _VC, _FC, _VC, MAT_TYPE

#define LINEARNB(neurons) net.Add<mlpack::LinearNoBiasType<MAT_TYPE, mlpack::L2Regularizer>>(neurons, mlpack::L2Regularizer(hp.l2()))
#define LINEAR(neurons) net.Add<mlpack::LinearType<MAT_TYPE, mlpack::L2Regularizer>>(neurons, mlpack::L2Regularizer(hp.l2()))
#define ACTIVATION net.Add<mlpack::LeakyReLUType<MAT_TYPE>>()
#define DROPOUT net.Add<mlpack::DropoutType<MAT_TYPE>>(hp.dropout())

void PhonemeModel::setHyperparameters(Hyperparameters hp) {
    this->hp = hp;
}

void PhonemeModel::initModel() {
    net = NETWORK_TYPE();

    net.Add<mlpack::ConvolutionType<CONVT>>(
        128, // maps
        2,   // kernelWidth
        2,   // kernelHeight
        1,   // strideWidth
        1    // strideHeight
    );
    net.Add<mlpack::ConvolutionType<CONVT>>(
        64,  // maps
        2,   // kernelWidth
        2,   // kernelHeight
        2,   // strideWidth
        2    // strideHeight
    );
    ACTIVATION;

    DROPOUT;
    net.Add<mlpack::ConvolutionType<CONVT>>(
        128, // maps
        2,   // kernelWidth
        2,   // kernelHeight
        1,   // strideWidth
        1    // strideHeight
    );
    net.Add<mlpack::ConvolutionType<CONVT>>(
        64,  // maps
        2,   // kernelWidth
        2,   // kernelHeight
        2,   // strideWidth
        2    // strideHeight
    );
    ACTIVATION;

    DROPOUT;
    net.Add<mlpack::ConvolutionType<CONVT>>(
        128, // maps
        2,   // kernelWidth
        2,   // kernelHeight
        1,   // strideWidth
        1    // strideHeight
    );
    net.Add<mlpack::ConvolutionType<CONVT>>(
        64,  // maps
        2,   // kernelWidth
        2,   // kernelHeight
        2,   // strideWidth
        2    // strideHeight
    );
    ACTIVATION;

    DROPOUT;
    LINEAR(1024);
    ACTIVATION;

    DROPOUT;
    LINEAR(1024);
    ACTIVATION;

    DROPOUT;
    LINEAR(outputSize);
    net.Add<mlpack::LogSoftMaxType<MAT_TYPE>>();

    net.InputDimensions() = { FFT_FRAMES, FRAME_SIZE, 3 };
}

void PhonemeModel::initOptimizer() {
    /* Adam initalization
    optim = OPTIMIZER_TYPE(
        0,      // Step size of the optimizer.
        0,      // Batch size. Number of data points that are used in each iteration.
        0.9,    // Exponential decay rate for the first moment estimates.
        0.999,  // Exponential decay rate for the weighted infinity norm estimates.
        1e-8,   // Value used to initialise the mean squared gradient parameter.
        0,      // Max number of iterations.
        1e-8,   // Tolerance.
        true);
    //*/
    //* AdaBelief initalization
    optim = OPTIMIZER_TYPE(
        0,      // Step size for each iteration.
        0,      // Number of points to process in a single step.
        0.9,    // The exponential decay rate for the 1st moment estimates.
        0.999,  // The exponential decay rate for the 2nd moment estimates.
        1e-8,   // A small constant for numerical stability.
        0,      // Maximum number of iterations allowed (0 means no limit).
        1e-8,   // Maximum absolute tolerance to terminate algorithm.
        true);
    //*/
    /* StandardSGD initalization
    optim = OPTIMIZER_TYPE(
        0,      // Step size for each iteration.
        0,      // Number of points to process in a single step.
        1e-8,   // Maximum absolute tolerance to terminate algorithm.
        true);
    //*/

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
    } else if (logger.has_value()) {
        logger->log(std::format("{} does not exist", ARCHIVE_FILE), Logger::WARNING);
    }

    outputSize = G_PS.size();

    config = Config(CONFIG_FILE, CURRENT_VERSION);
    config.setDefault("input_features", inputSize)
        .setDefault("output_features", outputSize)
        .setDefault("sample_rate", sampleRate);
    config.load();

    if (!config.matchesDefault()) {
        config.useDefault();
        loaded = false;
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
