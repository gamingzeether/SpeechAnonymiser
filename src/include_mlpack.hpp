#ifdef USE_GPU
  #define MLPACK_HAS_COOT
  #define CL_TARGET_OPENCL_VERSION 300
  #include <bandicoot>
#endif

#define ARMA_DONT_PRINT_FAST_MATH_WARNING
#include <mlpack.hpp>
#include "Classifier/negative_log_likelihood_w.hpp"
