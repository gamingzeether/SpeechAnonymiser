#include "PhonemeModel.h"

void PhonemeModel::initModel(Hyperparameters hp) {
    int outputSize = ClassifierHelper::instance().inversePhonemeSet.size();

    net = NETWORK_TYPE();

    net.Add<mlpack::LinearNoBiasType<MAT_TYPE, mlpack::L2Regularizer>>(1024, mlpack::L2Regularizer(hp.l2()));
    net.Add<mlpack::GELUType<MAT_TYPE>>();
    net.Add<mlpack::DropoutType<MAT_TYPE>>(hp.dropout());

    net.Add<mlpack::LinearType<MAT_TYPE, mlpack::L2Regularizer>>(1024, mlpack::L2Regularizer(hp.l2()));
    net.Add<mlpack::GELUType<MAT_TYPE>>();
    net.Add<mlpack::DropoutType<MAT_TYPE>>(hp.dropout());

    net.Add<mlpack::LinearType<MAT_TYPE, mlpack::L2Regularizer>>(1024, mlpack::L2Regularizer(hp.l2()));
    net.Add<mlpack::LeakyReLUType<MAT_TYPE>>();
    net.Add<mlpack::DropoutType<MAT_TYPE>>(hp.dropout());

    net.Add<mlpack::LinearType<MAT_TYPE, mlpack::L2Regularizer>>(768, mlpack::L2Regularizer(hp.l2()));
    net.Add<mlpack::LeakyReLUType<MAT_TYPE>>();
    net.Add<mlpack::DropoutType<MAT_TYPE>>(hp.dropout());

    net.Add<mlpack::LinearType<MAT_TYPE, mlpack::L2Regularizer>>(768, mlpack::L2Regularizer(hp.l2()));
    net.Add<mlpack::LeakyReLUType<MAT_TYPE>>();
    net.Add<mlpack::DropoutType<MAT_TYPE>>(hp.dropout());

    net.Add<mlpack::LinearType<MAT_TYPE, mlpack::L2Regularizer>>(outputSize, mlpack::L2Regularizer(hp.l2()));
    net.Add<mlpack::LogSoftMaxType<MAT_TYPE>>();

    optim = ens::Adam(
        STEP_SIZE,  // Step size of the optimizer.
        0, // Batch size. Number of data points that are used in each iteration.
        0.9,        // Exponential decay rate for the first moment estimates.
        0.999, // Exponential decay rate for the weighted infinity norm estimates.
        1e-8,  // Value used to initialise the mean squared gradient parameter.
        0, // Max number of iterations.
        1e-8,           // Tolerance.
        true);

    optim.BatchSize() = hp.batchSize();
    optim.StepSize() = hp.stepSize();
}
