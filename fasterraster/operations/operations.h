#pragma once

#include <stdint.h>

void HILLSHADE_MP(const float *dem, uint8_t *hs, const int AZIMUTH, const int ALITITUDE, const float CELL_SIZE, const int NROWS, const int NCOLS, const int NUMT);

void HILLSHADE_MP_FASTER(const float *dem, uint8_t *hs, const int AZIMUTH, const int ALITITUDE, const float CELL_SIZE, const int NROWS, const int NCOLS, const int NUMT);

void SLOPE_MP(const float *dem, float *slope, const float CELL_SIZE, const int NROWS, const int NCOLS, const int NUMT);

void HILLSHADE_SLOPE_MP(const float *dem, uint8_t *hs, float *slope, const int AZIMUTH, const int ALITITUDE, const float CELL_SIZE, const int NROWS, const int NCOLS, const int NUMT);

void ASPECT_MP(const float *dem, float *aspect, const float CELL_SIZE, const int NROWS, const int NCOLS, const int NUMT);

void STATIC_CENTER_CROP_MP(float* crop, const float* full, const int crop_NROWS, const int crop_NCOLS, const int full_NROWS, const int full_NCOLS, const int NUMT);

void STATIC_RANDOM_CROP_MP(float* crop, const float* full, const int crop_NROWS, const int crop_NCOLS, const int full_NROWS, const int full_NCOLS, const int NUMT);

void MULTIPY_SCALER_MP(float* arr, const float SCALER, const int SIZE, const int NUMT);

void MULTIPY_SCALER_SSE(float* arr, const float SCALER, const int SIZE, const int NUMT);
