#pragma once 

#include "../common_inc.hpp"

#include <functional>
#include <armadillo>

// Class for custom mlpack training callbacks
template<typename MatType = arma::mat>
class TrainingExecType {
public:
    /**
     * Set up the early stop at min loss class, which keeps track of the minimum
     * loss and stops the optimization process if the loss stops decreasing.
     *
     * @param func, callback to return immediate loss evaluated by the function
     * @param patienceIn The number of epochs to wait after the minimum loss has
     *    been reached or no improvement has been made (Default: 10).
     */
    TrainingExecType(
        std::function<void(size_t)> func)
      : numEpochs(0),
        epochFunc(func)
    {
        // Nothing to do here
    }

    /**
     * Callback function called at the end of a pass over the data.
     *
     * @param optimizer The optimizer used to update the function.
     * @param function Function to optimize.
     * @param coordinates Starting point.
     * @param epoch The index of the current epoch.
     * @param objective Objective value of the current point.
     */
    template<typename OptimizerType, typename FunctionType>
    bool EndEpoch(OptimizerType& /* optimizer */,
                  FunctionType& /* function */,
                  const MatType& /* coordinates */,
                  const size_t /* epoch */,
                  double /* objective */)
    {
        epochFunc(numEpochs++);
        return false;
    }
private:
    size_t numEpochs;
    std::function<void(size_t)> epochFunc;
};