#pragma once
#include <stdint.h>

void construct_flo(float* flo, const float* u, const float* v, const int SIZE);

void get_u_v(float* u, float* v, const float* flo, const int SIZE, const int NUMT);

void write_flo(const float* flo, const char* filename, const int NROWS, const int NCOLS);

void write_flo_MP(const float* u, const float* v, const char* fbase, const int NUMF, const int NROWS, const int NCOLS, const int START_IDX, const int NUMT);

void read_flo(float* flo, const char* filename, const int NROWS, const int NCOLS);

void multiply_flo_scalers(float* cpy_flo, const float* src_flo, const int ITERS, const int NROWS, const int NCOLS, const int NUMT);

void multiply_flo_mask_arr(float* arr1, const uint8_t* arr2, const int SIZE, const int NUMT);

void multiply_flo_and_save(const float* u, const float* v, const char* filename, const int iters, const int NROWS, const int NCOLS, const int NUMT);