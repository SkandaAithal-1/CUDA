#pragma once

__global__ void vector_addition(const float *a, const float *b, float *c, int N);

__global__ void matrix_addition(const float *A, const float *B, float *C, int M, int N);

__global__ void matrix_vec_mult(const float *A, const float *x, float *y, int M, int N);

__global__ void vector_sum(float *x, int N);