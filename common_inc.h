#pragma once

#define MLPACK_ANN_IGNORE_SERIALIZATION_WARNING
#define ARMA_PRINT_EXCEPTIONS
#define OUT

#ifdef __SANITIZE_ADDRESS__
#define _DISABLE_VECTOR_ANNOTATION
#define _DISABLE_STRING_ANNOTATION
#endif

#include <cmath>
#define VALID_FLOAT(v) if (!std::isfinite(v)) throw("fuck");
