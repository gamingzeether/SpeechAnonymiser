#pragma once

#include <assert.h>

#define MLPACK_ANN_IGNORE_SERIALIZATION_WARNING
#define ARMA_PRINT_EXCEPTIONS
#define OUT

#ifdef __SANITIZE_ADDRESS__
#define _DISABLE_VECTOR_ANNOTATION
#define _DISABLE_STRING_ANNOTATION
#endif

#define QUOTE(v) #v

#define STRINGIFY(v) QUOTE(v)

#define CPU_CUBE_TYPE arma::fcube
#define CPU_MAT_TYPE arma::fmat

#ifdef USE_GPU
  #define GPU_CUBE_TYPE coot::fcube
  #define GPU_MAT_TYPE coot::fmat

  #define CUBE_TYPE GPU_CUBE_TYPE
  #define MAT_TYPE GPU_MAT_TYPE
#else
  #define CUBE_TYPE CPU_CUBE_TYPE
  #define MAT_TYPE CPU_MAT_TYPE
#endif
