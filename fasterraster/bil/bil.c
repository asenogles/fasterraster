#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>

#define NBANDS 1

/* This bil implementation currently does not support multi band rasters */

void write_hdr(const char* filename, const int NROWS, const int NCOLS, const double ULXMAP, const double ULYMAP, const float XDIM, const float YDIM, const float NODATA) {

    // open file
    FILE *stream = fopen(filename, "wb");
    //FILE *stream;
    //fopen_s(&stream, filename, "wb");
    if (stream == 0) {
        printf("filname: %s is invalid, writing file failed!\n", filename);
    }

    
    fprintf(stream, "BYTEORDER      I\n");
    fprintf(stream, "LAYOUT         BIL\n");
    fprintf(stream, "NROWS          %d\n", NROWS);
    fprintf(stream, "NCOLS          %d\n", NCOLS);
    fprintf(stream, "NBANDS         %d\n", NBANDS);
    fprintf(stream, "NBITS          %zd\n", sizeof(float) * 8);
    fprintf(stream, "BANDROWBYTES   %zd\n", sizeof(float) * NROWS);
    fprintf(stream, "TOTALROWBYTES  %zd\n", sizeof(float) * NROWS * NBANDS);
    fprintf(stream, "PIXELTYPE      FLOAT\n");
    fprintf(stream, "ULXMAP         %0.3f\n", ULXMAP);
    fprintf(stream, "ULYMAP         %0.3f\n", ULYMAP);
    fprintf(stream, "XDIM           %0.3f\n", XDIM);
    fprintf(stream, "YDIM           %0.3f\n", YDIM);
    fprintf(stream, "NODATA         %0.2f\n", NODATA);

    fclose(stream); 
}

void write_bil(const float* dem, const char* filename, const int NROWS, const int NCOLS) {

    // open file
    FILE *stream = fopen(filename, "wb");
    //FILE *stream;
    //fopen_s(&stream, filename, "wb");
    if (stream == 0) {
        printf("filname: %s is invalid, writing file failed!\n", filename);
    }
    
    // write data flattened
    for (int r=0; r < NROWS; r++) {
        fwrite(&dem[r*NCOLS*NBANDS], sizeof(float), NCOLS, stream);
    }
    fclose(stream); 
}

void read_bil(float* data, const char* fname, const int NROWS, const int NCOLS) {

    FILE *stream = fopen(fname, "rb");
    if (stream == 0) {
        printf("filname: %s is invalid, writing file failed!\n", fname);
    }
        
    for (int r=0; r < NROWS; r++) {
        fread(&data[r*NCOLS], sizeof(float), NCOLS, stream);
    }
}