#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>

#define SSE_WIDTH 4
#define PI 3.14159265358979323846f


void HILLSHADE_MP(const float *dem, uint8_t *hs, const int AZIMUTH, const int ALITITUDE, const float CELL_SIZE, const int NROWS, const int NCOLS, const int NUMT) {
    /* OpenMP implementation of the hillshade function */

    // declare consts
    const float AZIMUTH_RAD = (float)(AZIMUTH % 180) * PI / 180.0f; // assign azimuth in radians
    const float ALTITUDE_RAD = (float)ALITITUDE * PI / 180.0f; // assign altitude in radians
    const float SIN_ALTITUDE = sinf(ALTITUDE_RAD);
    const float COS_ALTITUDE = cosf(ALTITUDE_RAD);
    omp_set_num_threads(NUMT);
    int r;
    #pragma omp parallel for default(none) shared(NROWS, NCOLS, CELL_SIZE, AZIMUTH_RAD, SIN_ALTITUDE, COS_ALTITUDE, dem, hs)
    for (r=0; r < NROWS; r++) {
        int i = r * NCOLS;
        int im = (r-1) * NCOLS;
        int ip = (r+1) * NCOLS;
        for (int c=0; c < NCOLS; c++) {
            int j = c;
            int jm = (c-1);
            int jp = (c+1);
            
            // compute y gradients
            float y;
            if (r == 0) {
                y = (dem[ip + j] / CELL_SIZE) - (dem[i + j] / CELL_SIZE);
            }
            else if (r == NROWS-1) {
                y = (dem[i + j] / CELL_SIZE) - (dem[im + j] / CELL_SIZE);
            }
            else {
                y = ((dem[ip + j] / CELL_SIZE) - (dem[im + j] / CELL_SIZE)) / 2.0f;
            }

            float x;
            if (c == 0) {
                x = (dem[i + jp] / CELL_SIZE) - (dem[i + j] / CELL_SIZE);
            }
            else if (c == NCOLS-1) {
                x = (dem[i + j] / CELL_SIZE) - (dem[i + jm] / CELL_SIZE);
            }
            else {
                x = ((dem[i + jp] / CELL_SIZE) - (dem[i + jm] / CELL_SIZE)) / 2.0f;
            }

            // compute hillshade
            float slope = PI / 2.0f - atanf(sqrtf(x * x + y * y));
            float aspect = atan2f(-y, x);
            float shaded = SIN_ALTITUDE * sinf(slope) + COS_ALTITUDE * cosf(slope) * cosf((AZIMUTH_RAD - PI / 2.0f) - aspect);
            hs[i+j] = (uint8_t)(255 * (shaded + 1) / 2);
        }
    }
}

void HILLSHADE_MP_FASTER(const float *dem, uint8_t *hs, const int AZIMUTH, const int ALITITUDE, const float CELL_SIZE, const int NROWS, const int NCOLS, const int NUMT) {
    /* OpenMP implementation of the hillshade function ~ faster, 
    see - https://observablehq.com/@sahilchinoy/a-faster-hillshader*/

    // declare consts
    const float AZIMUTH_RAD = (float)(AZIMUTH % 180) * PI / 180.0f; // assign azimuth in radians
    const float ALTITUDE_RAD = (float)ALITITUDE * PI / 180.0f; // assign altitude in radians
    const float A1 = sinf(ALTITUDE_RAD);
    const float A2 = sinf(AZIMUTH_RAD) * cosf(ALTITUDE_RAD);
    const float A3 = cosf(AZIMUTH_RAD)* cosf(ALTITUDE_RAD);

    omp_set_num_threads(NUMT);
    int r;
    #pragma omp parallel for default(none) shared(NROWS, NCOLS, CELL_SIZE, A1, A2, A3, dem, hs)
    for (r=0; r < NROWS; r++) {
        int i = r * NCOLS;
        int im = (r-1) * NCOLS;
        int ip = (r+1) * NCOLS;
        for (int c=0; c < NCOLS; c++) {
            int j = c;
            int jm = (c-1);
            int jp = (c+1);
            
            // compute y gradient
            float y;
            if (r == 0) {
                y = (dem[ip + j] / CELL_SIZE) - (dem[i + j] / CELL_SIZE);
            }
            else if (r == NROWS-1) {
                y = (dem[i + j] / CELL_SIZE) - (dem[im + j] / CELL_SIZE);
            }
            else {
                y = ((dem[ip + j] / CELL_SIZE) - (dem[im + j] / CELL_SIZE)) / 2.0f;
            }

            // compute x gradient
            float x;
            if (c == 0) {
                x = (dem[i + jp] / CELL_SIZE) - (dem[i + j] / CELL_SIZE);
            }
            else if (c == NCOLS-1) {
                x = (dem[i + j] / CELL_SIZE) - (dem[i + jm] / CELL_SIZE);
            }
            else {
                x = ((dem[i + jp] / CELL_SIZE) - (dem[i + jm] / CELL_SIZE)) / 2.0f;
            }

            // calculate hillshade
            float L = (A1 - (x*A2) - (y*A3)) / sqrtf(1.0f + (x*x) + (y*y));
            hs[i+j] = (uint8_t)(255 * (L + 1) / 2);
        }
    }
}

void SLOPE_MP(const float *dem, float *slope, const float CELL_SIZE, const int NROWS, const int NCOLS, const int NUMT) {

    omp_set_num_threads(NUMT);
    int r;
    #pragma omp parallel for default(none) shared(NROWS, NCOLS, CELL_SIZE, dem, slope)
    for (r=0; r < NROWS; r++) {
        int i = r * NCOLS;
        int im = (r-1) * NCOLS;
        int ip = (r+1) * NCOLS;
        for (int c=0; c < NCOLS; c++) {
            int j = c;
            int jm = c - 1;
            int jp = c + 1;
            
            // compute y gradients
            float y;
            if (r == 0) {
                y = (dem[ip + j] / CELL_SIZE) - (dem[i + j] / CELL_SIZE);
            }
            else if (r == NROWS-1) {
                y = (dem[i + j] / CELL_SIZE) - (dem[im + j] / CELL_SIZE);
            }
            else {
                y = ((dem[ip + j] / CELL_SIZE) - (dem[im + j] / CELL_SIZE)) / 2.0f;
            }

            float x;
            if (c == 0) {
                x = (dem[i + jp] / CELL_SIZE) - (dem[i + j] / CELL_SIZE);
            }
            else if (c == NCOLS-1) {
                x = (dem[i + j] / CELL_SIZE) - (dem[i + jm] / CELL_SIZE);
            }
            else {
                x = ((dem[i + jp] / CELL_SIZE) - (dem[i + jm] / CELL_SIZE)) / 2.0f;
            }

            // compute slope
            slope[i+j] = PI / 2.0f - atanf(sqrtf(x * x + y * y));
        }
    }
}

void HILLSHADE_SLOPE_MP(const float *dem, uint8_t *hs, float *slope, const int AZIMUTH, const int ALITITUDE, const float CELL_SIZE, const int NROWS, const int NCOLS, const int NUMT) {
    /* calculate slope & hillshade at same time */

    // declare consts
    const float AZIMUTH_RAD = (float)(AZIMUTH % 180) * PI / 180.0f; // assign azimuth in radians
    const float ALTITUDE_RAD = (float)ALITITUDE * PI / 180.0f; // assign altitude in radians
    const float A1 = sinf(ALTITUDE_RAD);
    const float A2 = sinf(AZIMUTH_RAD) * cosf(ALTITUDE_RAD);
    const float A3 = cosf(AZIMUTH_RAD)* cosf(ALTITUDE_RAD);

    omp_set_num_threads(NUMT);
    int r;
    #pragma omp parallel for default(none) shared(NROWS, NCOLS, CELL_SIZE, A1, A2, A3, dem, hs, slope)
    for (r=0; r < NROWS; r++) {
        int i = r * NCOLS;
        int im = (r-1) * NCOLS;
        int ip = (r+1) * NCOLS;
        for (int c=0; c < NCOLS; c++) {
            int j = c;
            int jm = c - 1;
            int jp = c + 1;
            
            // compute y gradient
            float y;
            if (r == 0) {
                y = (dem[ip + j] / CELL_SIZE) - (dem[i + j] / CELL_SIZE);
            }
            else if (r == NROWS-1) {
                y = (dem[i + j] / CELL_SIZE) - (dem[im + j] / CELL_SIZE);
            }
            else {
                y = ((dem[ip + j] / CELL_SIZE) - (dem[im + j] / CELL_SIZE)) / 2.0f;
            }
            
            // compute x gradient
            float x;
            if (c == 0) {
                x = (dem[i + jp] / CELL_SIZE) - (dem[i + j] / CELL_SIZE);
            }
            else if (c == NCOLS-1) {
                x = (dem[i + j] / CELL_SIZE) - (dem[i + jm] / CELL_SIZE);
            }
            else {
                x = ((dem[i + jp] / CELL_SIZE) - (dem[i + jm] / CELL_SIZE)) / 2.0f;
            }

            // compute hillshade / slope
            slope[i+j] = PI / 2.0f - atanf(sqrtf(x * x + y * y));
            float L = (A1 - (x*A2) - (y*A3)) / sqrtf(1.0f + (x*x) + (y*y));
            hs[i+j] = (uint8_t)(255 * (L + 1) / 2);
        }
    }
}

void ASPECT_MP(const float *dem, float *aspect, const float CELL_SIZE, const int NROWS, const int NCOLS, const int NUMT) {

    omp_set_num_threads(NUMT);
    int r;
    #pragma omp parallel for default(none) shared(NROWS, NCOLS, CELL_SIZE, dem, aspect)
    for (r=0; r < NROWS; r++) {
        int i = r * NCOLS;
        int im = (r-1) * NCOLS;
        int ip = (r+1) * NCOLS;
        for (int c=0; c < NCOLS; c++) {
            int j = c;
            int jm = c - 1;
            int jp = c + 1;
            
            // compute y gradients
            float y;
            if (r == 0) {
                y = (dem[ip + j] / CELL_SIZE) - (dem[i + j] / CELL_SIZE);
            }
            else if (r == NROWS-1) {
                y = (dem[i + j] / CELL_SIZE) - (dem[im + j] / CELL_SIZE);
            }
            else {
                y = ((dem[ip + j] / CELL_SIZE) - (dem[im + j] / CELL_SIZE)) / 2.0f;
            }

            float x;
            if (c == 0) {
                x = (dem[i + jp] / CELL_SIZE) - (dem[i + j] / CELL_SIZE);
            }
            else if (c == NCOLS-1) {
                x = (dem[i + j] / CELL_SIZE) - (dem[i + jm] / CELL_SIZE);
            }
            else {
                x = ((dem[i + jp] / CELL_SIZE) - (dem[i + jm] / CELL_SIZE)) / 2.0f;
            }

            // compute aspect
            aspect[i+j] = atan2f(-x, y);
        }
    }
}

void STATIC_CENTER_CROP_MP(float* crop, const float* full, const int crop_NROWS, const int crop_NCOLS, const int full_NROWS, const int full_NCOLS, const int NUMT) {

    const int start_row_idx = (full_NROWS - crop_NROWS) / 2;
    const int start_col_idx = (full_NCOLS - crop_NCOLS) / 2;

    omp_set_num_threads(NUMT);
    int r;
    #pragma omp parallel for default(none) shared(crop_NROWS, crop_NCOLS, start_row_idx, start_col_idx, full_NCOLS, crop, full)
    for (r=0; r < crop_NROWS; r++) {
        int i_c = r * crop_NCOLS;
        int i_f = (r+start_row_idx) * full_NCOLS;
        for (int c=0; c < crop_NCOLS; c++) {
            int j_c = c;
            int j_f = c + start_col_idx;
            crop[i_c + j_c] = full[i_f + j_f];
        }
    }
}

void STATIC_RANDOM_CROP_MP(float* crop, const float* full, const int crop_NROWS, const int crop_NCOLS, const int full_NROWS, const int full_NCOLS, const int NUMT) {

    // seed random num generator - don't call more than once per second
    static int last_call = 0;
    
    if ( last_call != time(0) ) {
        srand((unsigned int)time(0));
        last_call = time(0);
    }

    // get random starting index
    const int start_row_idx = rand() / RAND_MAX * (full_NROWS - crop_NROWS);
    const int start_col_idx = rand() / RAND_MAX * (full_NCOLS - crop_NCOLS);

    omp_set_num_threads(NUMT);
    int r;
    #pragma omp parallel for default(none) shared(crop_NROWS, crop_NCOLS, start_row_idx, start_col_idx, full_NCOLS, crop, full)
    for (r=0; r < crop_NROWS; r++) {
        int i_c = r * crop_NCOLS;
        int i_f = (r+start_row_idx) * full_NCOLS;
        for (int c=0; c < crop_NCOLS; c++) {
            int j_c = c;
            int j_f = c + start_col_idx;
            crop[i_c + j_c] = full[i_f + j_f];
        }
    }
}

void MULTIPY_SCALER_MP(float* arr, const float SCALER, const int SIZE, const int NUMT) {
    omp_set_num_threads(NUMT);
    int i;
    #pragma omp parallel for default(none) shared(SIZE, SCALER, arr)
    for (i=0; i < SIZE; i++) {
        arr[i]*= SCALER;
    }
}

void MULTIPY_SCALER_SSE(float* arr, const float SCALER, const int SIZE, const int NUMT) {
    
    omp_set_num_threads(NUMT);
    const int SIZE_LIMIT = (SIZE / SSE_WIDTH) * SSE_WIDTH;
    const __m128 S = _mm_load_ps1(&SCALER);
    const register float* p_arr = arr;
    
    int i;
    #pragma omp parallel for default(none) shared(SIZE_LIMIT, S, arr, p_arr)
    for (i=0; i<SIZE_LIMIT; i+=SSE_WIDTH) {
        _mm_storeu_ps(&arr[i], _mm_mul_ps(_mm_loadu_ps(&p_arr[i]), S));
    }
    for (i=SIZE_LIMIT; i < SIZE; i++) {
        arr[i]*= SCALER;
    }
}
