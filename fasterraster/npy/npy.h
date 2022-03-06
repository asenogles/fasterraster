#pragma once
#include <stdint.h>


int numPlaces (int n);

void write_u8bit_npy(uint8_t* data, const char* filename, const int NROWS, const int NCOLS);

void write_float32_npy(float* data, const char* filename, const int NROWS, const int NCOLS);