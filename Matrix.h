#pragma once
#include "define.h"

class Matrix {
public:
	Matrix() : rows(1), columns(1) { values = new float[1]; };
	Matrix(size_t r, size_t c) : rows(r), columns(c) { values = new float[r * c]; };

	inline void set(size_t r, size_t c, float val) { values[c + r * columns] = val; };
	inline float get(size_t r, size_t c) const { return values[c + r * columns]; }

	Matrix add(const Matrix& other);
	Matrix scalar_multiply(float scalar);
	Matrix transpose();
	const Matrix matrix_multiply(const Matrix& other);
	const void matrix_multiply(const Matrix& other, Matrix& out);
private:
	size_t rows;
	size_t columns;
	float* values;
};
