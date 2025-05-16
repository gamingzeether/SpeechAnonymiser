// Include linear algebra libraries
// Armadillo (and Bandicoot if enabled)

#define ARMA_DONT_PRINT_FAST_MATH_WARNING

#include <armadillo>

#ifdef USE_GPU
  #define MLPACK_HAS_COOT
  #define CL_TARGET_OPENCL_VERSION 300
  #include <bandicoot>
#endif
