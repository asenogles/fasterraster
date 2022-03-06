#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>

#define SSE_WIDTH 4
#define N_FLO_BANDS 2
#define TAG_STRING "PIEH"


void construct_flo(float* flo, const float* u, const float* v, const int SIZE) {
    for (int i=0; i < SIZE; i++) {
        flo[i * N_FLO_BANDS] = u[i];
        flo[(i * N_FLO_BANDS) + 1] = v[i];
    }
}

void get_u_v(float* u, float* v, const float* flo, const int SIZE, const int NUMT) {
    /* Copy flo array to seperate u/v arrays, size is size of u/v array */
    omp_set_num_threads(NUMT);
    int i;
    #pragma omp parallel for default(none) shared(SIZE, u, v, flo)
    for (i=0; i < SIZE; i++) {
        u[i] = flo[i * N_FLO_BANDS];
        v[i] = flo[(i * N_FLO_BANDS) + 1];
    }
}

void write_flo(const float* flo, const char* filename, const int NROWS, const int NCOLS) {

    // open file
    FILE *stream = fopen(filename, "wb");
    if (stream == 0) {
        printf("filname: %s is invalid, writing file failed!\n", filename);
    }

    // write header
    fprintf(stream, TAG_STRING); // write tag - "PIEH" in ASCII
    fwrite(&NCOLS, sizeof(int), 1, stream); // write width as 32 bit int
    fwrite(&NROWS, sizeof(int), 1, stream); // write row as 32 bit int
    
    // write data flattened - u[0,0],v[0,0],u[0,1],v[0,1].....
    int n = NCOLS * N_FLO_BANDS;
    for (int r=0; r < NROWS; r++) {
        fwrite(&flo[r*NCOLS*N_FLO_BANDS], sizeof(float), n, stream);
    }
    fclose(stream);
}

void write_flo_MP(const float* u, const float* v, const char* fbase, const int NUMF, const int NROWS, const int NCOLS, const int START_IDX, const int NUMT) {

    // Set # of threads to use
    if (NUMF < NUMT)
        omp_set_num_threads(NUMF);
    else
        omp_set_num_threads(NUMT);

    const int SIZE = NROWS * NCOLS;
    const int F_SIZE = (int)strlen(fbase);

    int i;
    #pragma omp parallel for default(none) shared(NUMF, SIZE, F_SIZE, START_IDX, NROWS, NCOLS, u, v, fbase)
    for (i=0; i < NUMF; i++) {

        // construct flo array
        float* flo = (float*)malloc(sizeof(float) * N_FLO_BANDS * SIZE);
        construct_flo(flo, u + (i*SIZE), v + (i*SIZE), SIZE);

        // now save array as flo
        char* filename = (char*)malloc(sizeof(char) * (F_SIZE + 9));

        #if defined(__linux__)
            snprintf(filename, F_SIZE+1, "%s", fbase);
        #elif defined(_WIN32)
            snprintf(filename, F_SIZE, fbase);
        #endif
        
        const int idx = START_IDX + i;
        if (idx < 10)
            snprintf(filename + F_SIZE, 9, "000%d.flo", idx);
        else if (idx < 100)
            snprintf(filename + F_SIZE, 9, "00%d.flo", idx);
        else if (idx < 1000)
            snprintf(filename + F_SIZE, 9, "0%d.flo", idx); 
        else
            snprintf(filename + F_SIZE, 9, "%d.flo", idx);

        write_flo(flo, filename, NROWS, NCOLS);
        free(flo);
        free(filename);
    }
}

void read_flo(float* flo, const char* filename, const int NROWS, const int NCOLS) {

    FILE *stream = fopen(filename, "rb");
    if (stream == 0) {
        printf("filname: %s is invalid, writing file failed!\n", filename);
    }

    // read header
    char header[4];
    //float* flo;
    fscanf(stream, "%4c", header);
    //printf("header: %s\n", header);
    if (strcmp(header, TAG_STRING)) {
        int ncols, nrows;
        fread(&ncols, sizeof(int), 1, stream);
        fread(&nrows, sizeof(int), 1, stream);

        //flo = (float*)malloc(sizeof(float) * N_FLO_BANDS * nrows * ncols);
        if (nrows == NROWS && ncols == NCOLS) {
            const int N = NCOLS * N_FLO_BANDS;
            for (int r=0; r < NROWS; r++) {
                fread(&flo[r * N], sizeof(float), N, stream);
            }
        }
        else {
            printf("dims provided are incorrect NROWS provided: %d, nrows read: %d, NCOLS provided: %d, ncols read: %d\n", NROWS, nrows, NCOLS, ncols);
        }
    }
    else {
        printf("Invalid flo header tag: %s, should be: %s", header, TAG_STRING);
    }
}

void multiply_flo_scalers(float* cpy_flo, const float* src_flo, const int ITERS, const int NROWS, const int NCOLS, const int NUMT) {
    
    omp_set_num_threads(NUMT);
    
    const int SIZE = NROWS * NCOLS * N_FLO_BANDS;

    const int size_limit = (SIZE / SSE_WIDTH) * SSE_WIDTH;

    //register float *p_cpy = cpy_flo;
    const register float *p_src = src_flo;

    for (int t=0; t < ITERS; t++) {
        const int i = t * SIZE;
        const float scalerF = (float)t+1;
        const __m128 sf_ps = _mm_load_ps1(&scalerF);
        int j;
        #pragma omp parallel for default(none) shared(size_limit, sf_ps, i, cpy_flo, p_src)
        for (j=0; j < size_limit; j+=SSE_WIDTH) {
            __m128 src_ps = _mm_loadu_ps(&p_src[j]);
            _mm_storeu_ps(&cpy_flo[i+j], _mm_mul_ps(src_ps, sf_ps));
            //cpy_flo[i + j] = src_flo[j] * scalerF;
        }
        for (int j=size_limit; j < SIZE; j++) {
                cpy_flo[i + j] = src_flo[j] * scalerF;
            }
    }
}

void multiply_flo_mask_arr(float* arr1, const uint8_t* arr2, const int SIZE, const int NUMT) {
    /* multiply flo grid by 8bit binary mask */
    omp_set_num_threads(NUMT);
    int i;
    #pragma omp parallel for default(none) shared(SIZE, arr1, arr2)
    for (i=0; i < SIZE; i++) {
        arr1[i * N_FLO_BANDS]*= arr2[i];
        arr1[(i * N_FLO_BANDS) + 1]*= arr2[i];
    } 
}

void multiply_flo_and_save(const float* u, const float* v, const char* filename, const int iters, const int NROWS, const int NCOLS, const int NUMT) {
    
    if (iters < NUMT)
        omp_set_num_threads(iters);
    else
        omp_set_num_threads(NUMT);

    const int SIZE = NROWS * NCOLS;
    const int F_SIZE = (int)strlen(filename);

    int i;
    //#pragma omp parallel for default(none) shared(u, v, filename, SIZE, NROWS, NCOLS, F_SIZE)
    for (i=0; i < iters; i++) {
        // create and construct flo matrix
        float* flo = (float*)malloc(sizeof(float) * N_FLO_BANDS * SIZE);
        float scalerF = (float)i+1;
        for (int j=0; j < SIZE; j++) {
            flo[j*N_FLO_BANDS] = u[j] * scalerF;
            flo[(j*N_FLO_BANDS)+1] = v[j] * scalerF;
        }

        // now save array as flo TODO: make sure string is null terminated
        char* filename_iter = (char*)malloc(sizeof(char) * (F_SIZE + 7));
        snprintf(filename_iter, F_SIZE, "%s", filename);
        if (i < 10)
            snprintf(filename_iter + F_SIZE, 7, "0%d.flo", i);
        else
            snprintf(filename_iter + F_SIZE, 7, "%d.flo", i);        

        write_flo(flo, filename_iter, NROWS, NCOLS);
        
        free(filename_iter);
        free(flo);
    }
}