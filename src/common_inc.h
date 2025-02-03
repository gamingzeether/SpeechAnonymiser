#pragma once

#define MLPACK_ANN_IGNORE_SERIALIZATION_WARNING
#define ARMA_PRINT_EXCEPTIONS
#define OUT

#ifdef __SANITIZE_ADDRESS__
#define _DISABLE_VECTOR_ANNOTATION
#define _DISABLE_STRING_ANNOTATION
#endif

#define QUOTE(v) #v

//#define USE_GPU

#define CPU_MAT_TYPE arma::fmat
#ifdef USE_GPU
#define MAT_TYPE coot::fmat
#else
#define MAT_TYPE CPU_MAT_TYPE
#endif
