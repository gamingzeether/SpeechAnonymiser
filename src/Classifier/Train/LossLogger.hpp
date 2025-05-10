#pragma once 

#include "../../common_inc.hpp"

#include <fstream>
#include <armadillo>

template<typename MatType = MAT_TYPE>
class LossLogger {
public:
  LossLogger(std::string outFile, size_t interval = 1, bool seperateEpochs = true) :
             outCsv(outFile),
             flushInterval(interval),
             seperateEpochs(seperateEpochs),
             totalStepsTaken(0)
  {
    
  }

  // Start a new line for a new epoch
  template<typename OptimizerType, typename FunctionType>
  bool EndEpoch(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const size_t /* epoch */,
                double /* objective */)
  {
    if (seperateEpochs)
      outCsv << std::endl;
    return false;
  }

  // Calculate loss at each step
  template<typename OptimizerType, typename FunctionType>
  bool Evaluate(OptimizerType& optimizer,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const double objectiveIn)
  {
    totalStepsTaken++;
    double loss = objectiveIn / optimizer.BatchSize();
    outCsv << std::to_string(loss) << ",";
    if (totalStepsTaken % flushInterval == 0)
      outCsv << std::flush;
    return false;
  }

private:
  std::ofstream outCsv;
  bool seperateEpochs;
  size_t totalStepsTaken;
  size_t flushInterval;
};
