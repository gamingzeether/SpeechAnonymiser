#include "PhonemeModel.h"

#include <filesystem>
#include <fstream>
#include <zip.h>
#include "ModelSerializer.h"

#define ARCHIVE_FILE "classifier.zip"
#define MODEL_FILE "phoneme_model.bin"
#define CONFIG_FILE "classifier.json"
#define ZIP_FILES { MODEL_FILE, CONFIG_FILE }

#define CURRENT_VERSION -6

void PhonemeModel::setHyperparameters(Hyperparameters hp) {
    this->hp = hp;
}

void PhonemeModel::initModel() {
    net = NETWORK_TYPE();

    net.Add<mlpack::LinearType<MAT_TYPE, mlpack::L2Regularizer>>(2048, mlpack::L2Regularizer(hp.l2()));
    net.Add<mlpack::PReLUType<MAT_TYPE>>();
    net.Add<mlpack::DropoutType<MAT_TYPE>>(hp.dropout());

    net.Add<mlpack::LinearType<MAT_TYPE, mlpack::L2Regularizer>>(1536, mlpack::L2Regularizer(hp.l2()));
    net.Add<mlpack::PReLUType<MAT_TYPE>>();
    net.Add<mlpack::DropoutType<MAT_TYPE>>(hp.dropout());

    net.Add<mlpack::LinearType<MAT_TYPE, mlpack::L2Regularizer>>(1024, mlpack::L2Regularizer(hp.l2()));
    net.Add<mlpack::PReLUType<MAT_TYPE>>();
    net.Add<mlpack::DropoutType<MAT_TYPE>>(hp.dropout());

    net.Add<mlpack::LinearType<MAT_TYPE, mlpack::L2Regularizer>>(1024, mlpack::L2Regularizer(hp.l2()));
    net.Add<mlpack::PReLUType<MAT_TYPE>>();
    net.Add<mlpack::DropoutType<MAT_TYPE>>(hp.dropout());

    net.Add<mlpack::LinearType<MAT_TYPE, mlpack::L2Regularizer>>(768, mlpack::L2Regularizer(hp.l2()));
    net.Add<mlpack::PReLUType<MAT_TYPE>>();
    net.Add<mlpack::DropoutType<MAT_TYPE>>(hp.dropout());

    net.Add<mlpack::LinearType<MAT_TYPE, mlpack::L2Regularizer>>(768, mlpack::L2Regularizer(hp.l2()));
    net.Add<mlpack::PReLUType<MAT_TYPE>>();
    net.Add<mlpack::DropoutType<MAT_TYPE>>(hp.dropout());

    net.Add<mlpack::LinearType<MAT_TYPE, mlpack::L2Regularizer>>(768, mlpack::L2Regularizer(hp.l2()));
    net.Add<mlpack::PReLUType<MAT_TYPE>>();
    net.Add<mlpack::DropoutType<MAT_TYPE>>(hp.dropout());

    net.Add<mlpack::LinearType<MAT_TYPE, mlpack::L2Regularizer>>(outputSize, mlpack::L2Regularizer(hp.l2()));
    net.Add<mlpack::LogSoftMaxType<MAT_TYPE>>();
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

    optim.BatchSize() = hp.batchSize();
    optim.StepSize() = hp.stepSize();
}

void PhonemeModel::save(int checkpoint) {
    ModelSerializer::saveNetwork(MODEL_FILE, &net);
    config.save();

    int error = 0;
    zip_t* archive = zip_open(ARCHIVE_FILE, ZIP_CHECKCONS | ZIP_CREATE, &error);
    for (const char* string : ZIP_FILES) {
        zip_source_t* source = zip_source_file(archive, string, 0, ZIP_LENGTH_TO_END);
        error = zip_file_add(archive, string, source, ZIP_FL_OVERWRITE | ZIP_FL_ENC_UTF_8);
    }
    error = zip_close(archive);
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
        if (error != 0 && logger.has_value()) {
            logger->log(std::format("Zip file open error {}", error), Logger::WARNING);
        }
        size_t bufferSize = 64 * 1024 * 1024; // Allocate 64 MB
        char* buf = new char[bufferSize];
        for (const char* string : ZIP_FILES) {
            zip_file_t* file = zip_fopen(archive, string, ZIP_FL_UNCHANGED);
            zip_int64_t readBytes = zip_fread(file, buf, bufferSize);
            if (readBytes >= 0) {
                std::ofstream out = std::ofstream(string, std::ios::binary | std::ios::trunc);
                out.write(buf, readBytes);
                out.close();
            }
        }
        delete[] buf;
        zip_discard(archive);
    } else if (logger.has_value()) {
        logger->log(std::format("{} does not exist", ARCHIVE_FILE), Logger::WARNING);
    }

    outputSize = ClassifierHelper::instance().inversePhonemeSet.size();

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
