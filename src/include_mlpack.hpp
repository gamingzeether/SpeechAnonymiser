#ifdef USE_GPU
  #define MLPACK_HAS_COOT
  #include <bandicoot>
#endif

#define ARMA_DONT_PRINT_FAST_MATH_WARNING
#include <mlpack.hpp>
#include "negative_log_likelihood_w.hpp"
