#include "Matrix.h"

Matrix Matrix::add(const Matrix& other) {
#ifndef NDEBUG
	if (this->rows != other.rows || this->columns != other.columns) {
		throw;
	}
#endif

	Matrix new_matrix = Matrix(rows, columns);
	for (size_t i = 0; i < rows * columns; i++) {
		new_matrix.values[i] = this->values[i] + other.values[i];
	}
	return new_matrix;
}

Matrix Matrix::scalar_multiply(float scalar) {
	Matrix new_matrix = Matrix(rows, columns);
	for (size_t i = 0; i < rows * columns; i++) {
		new_matrix.values[i] = this->values[i] * scalar;
	}
	return new_matrix;
}

Matrix Matrix::transpose() {
	Matrix new_matrix = Matrix(rows, columns);
	for (size_t r = 0; r < this->rows; r++) {
		for (size_t c = 0; c < this->columns; c++) {
			new_matrix.set(r, c, this->get(r, c));
		}
	}
	return new_matrix;
}

const Matrix Matrix::matrix_multiply(const Matrix& other) {
	Matrix new_matrix = Matrix(this->rows, other.columns);
	matrix_multiply(other, new_matrix);
	return new_matrix;
}

const void Matrix::matrix_multiply(const Matrix& other, Matrix& out) {
#ifndef NDEBUG
	if (this->columns != other.rows) {
		throw;
	}
	if (out.rows != this->rows || out.columns != other.columns) {
		throw;
	}
#endif

	for (size_t r = 0; r < out.rows; r++) {
		for (size_t c = 0; c < out.columns; c++) {
			float sum = 0;
			for (size_t i = 0; i < this->columns; i++) {
				float a = this->get(r, i);
				float b = other.get(i, c);
				sum += a * b;
			}
			out.set(r, c, sum);
		}
	}
}
