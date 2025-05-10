#pragma once 

#include "../common_inc.hpp"

#include <functional>
#include <armadillo>

// Class for custom mlpack training callbacks
template<typename MatType = MAT_TYPE>
class TrainingExec {
public:
    TrainingExec(
        std::function<void(size_t, double)> epochFunc)
      : epochFunc(epochFunc)
    {
        // Nothing to do here
    }

    template<typename OptimizerType, typename FunctionType>
    bool EndEpoch(OptimizerType& /* optimizer */,
                  FunctionType& /* function */,
                  const MatType& /* coordinates */,
                  const size_t epoch,
                  double objective)
    {
        epochFunc(epoch, objective);
        return false;
    }
private:
    std::function<void(size_t, double)> epochFunc;
};