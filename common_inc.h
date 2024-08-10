#pragma once

#ifdef __SANITIZE_ADDRESS__
#define _DISABLE_VECTOR_ANNOTATION
#define _DISABLE_STRING_ANNOTATION
#endif

#define OUT

#include <cmath>
#define VALID_FLOAT(v) if (!std::isfinite(v)) throw("fuck");
