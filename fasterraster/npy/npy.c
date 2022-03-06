#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

int numPlaces (int n) {
    if (n < 0) n = (n == INT_MIN) ? INT_MAX : -n;
    if (n < 10) return 1;
    if (n < 100) return 2;
    if (n < 1000) return 3;
    if (n < 10000) return 4;
    if (n < 100000) return 5;
    if (n < 1000000) return 6;
    if (n < 10000000) return 7;
    if (n < 100000000) return 8;
    if (n < 1000000000) return 9;
    return 10;
}

void write_u8bit_npy(uint8_t* data, const char* filename, const int NROWS, const int NCOLS) {

    FILE *stream = fopen(filename, "wb");
    if (stream == 0) {
        printf("filname: %s is invalid, writing file failed!\n", filename);
    }

    // PRE header components
    const int PRE_LEN = 10;
    const unsigned char MJC_NUM = 0x93;
    const char MJC_STR[6] = "NUMPY";
    const unsigned char MJR_VER = 0x01;
    const unsigned char MNR_VER = 0x00;
    unsigned short hdr_len;
    
    // header string components
    const int DESCR_LEN = 17;
    const char DESCR[18] = "{'descr': '|u1', ";

    const int FOR_ORDR_LEN = 24;
    const char FOR_ORDR[25] = "'fortran_order': False, ";
    
    int shape_len = 16 + numPlaces(NROWS) + numPlaces(NCOLS);
    char* shape = (char*)malloc(sizeof(char) * shape_len);
    snprintf(shape, shape_len, "'shape': (%d, %d), }", NROWS, NCOLS);
    
    hdr_len = DESCR_LEN + FOR_ORDR_LEN + shape_len;
    int pad_len = 64 - ((PRE_LEN + hdr_len) % 64) - 1;
    hdr_len+= pad_len + 1;
    int total_hdr_len = PRE_LEN + hdr_len;


    char* padding = (char*)malloc(sizeof(char) * pad_len);
    for (int i=0; i < pad_len; i++) {
        padding[i] = 0x20;
    }

    const char NEWLINE[2] = "\n";

    char* npy_hdr = (char*)malloc(sizeof(char) * total_hdr_len);

    int offset = 0;
    memcpy(npy_hdr + offset, &MJC_NUM, 1);
    offset+=1;
    memcpy(npy_hdr + offset, MJC_STR, 5);
    offset+=5;
    memcpy(npy_hdr + offset, &MJR_VER, 1);
    offset+=1;
    memcpy(npy_hdr + offset, &MNR_VER, 1);
    offset+=1;
    memcpy(npy_hdr + offset, &hdr_len, 2);
    offset+=2;
    memcpy(npy_hdr + offset, DESCR, DESCR_LEN);
    offset+=DESCR_LEN;
    memcpy(npy_hdr + offset, FOR_ORDR, FOR_ORDR_LEN);
    offset+=FOR_ORDR_LEN;
    memcpy(npy_hdr + offset, shape, shape_len);
    offset+=shape_len;
    memcpy(npy_hdr + offset, padding, pad_len);
    offset+=pad_len;
    memcpy(npy_hdr + offset, NEWLINE, 1);

    printf("%d\n", offset);
    printf("%d, %d, %d, %d\n", DESCR_LEN, FOR_ORDR_LEN, shape_len, pad_len);

    // Now write data
    fwrite(npy_hdr, sizeof(char), total_hdr_len, stream); // write width as 32 bit int
    for (int i=0; i < NROWS; i++) {
        fwrite(&data[i*NCOLS], sizeof(uint8_t), NCOLS, stream); // write width as 32 bit int
    }
    //fwrite(data, sizeof(uint8_t), len, stream); // write width as 32 bit int
    fclose(stream); 

    free(shape);
    free(padding);
    free(npy_hdr);
}

void write_float32_npy(float* data, const char* filename, const int NROWS, const int NCOLS) {


    FILE *stream = fopen(filename, "wb");
    if (stream == 0) {
        printf("filname: %s is invalid, writing file failed!\n", filename);
    }

    // PRE header components
    const int PRE_LEN = 10;
    const unsigned char MJC_NUM = 0x93;
    const char MJC_STR[6] = "NUMPY";
    const unsigned char MJR_VER = 0x01;
    const unsigned char MNR_VER = 0x00;
    unsigned short hdr_len;
    
    // header string components
    const int DESCR_LEN = 17;
    const char DESCR[18] = "{'descr': '|f4', ";

    const int FOR_ORDR_LEN = 24;
    const char FOR_ORDR[25] = "'fortran_order': False, ";
    
    int shape_len = 16 + numPlaces(NROWS) + numPlaces(NCOLS);
    char* shape = (char*)malloc(sizeof(char) * shape_len);
    snprintf(shape, shape_len, "'shape': (%d, %d), }", NROWS, NCOLS);
    
    hdr_len = DESCR_LEN + FOR_ORDR_LEN + shape_len;
    int pad_len = 64 - ((PRE_LEN + hdr_len) % 64) - 1;
    hdr_len+= pad_len + 1;
    int total_hdr_len = PRE_LEN + hdr_len;
    

    char* padding = (char*)malloc(sizeof(char) * pad_len);
    for (int i=0; i < pad_len; i++) {
        padding[i] = 0x20;
    }

    const char NEWLINE[2] = "\n";

    char* npy_hdr = (char*)malloc(sizeof(char) * total_hdr_len);

    int offset = 0;
    memcpy(npy_hdr + offset, &MJC_NUM, 1);
    offset+=1;
    memcpy(npy_hdr + offset, MJC_STR, 5);
    offset+=5;
    memcpy(npy_hdr + offset, &MJR_VER, 1);
    offset+=1;
    memcpy(npy_hdr + offset, &MNR_VER, 1);
    offset+=1;
    memcpy(npy_hdr + offset, &hdr_len, 2);
    offset+=2;
    memcpy(npy_hdr + offset, DESCR, DESCR_LEN);
    offset+=DESCR_LEN;
    memcpy(npy_hdr + offset, FOR_ORDR, FOR_ORDR_LEN);
    offset+=FOR_ORDR_LEN;
    memcpy(npy_hdr + offset, shape, shape_len);
    offset+=shape_len;
    memcpy(npy_hdr + offset, padding, pad_len);
    offset+=pad_len;
    memcpy(npy_hdr + offset, NEWLINE, 1);

    //printf("%d\n", offset);
    //printf("%d, %d, %d, %d\n", DESCR_LEN, FOR_ORDR_LEN, shape_len, pad_len);

    // Now write data
    fwrite(npy_hdr, sizeof(char), total_hdr_len, stream); // write width as 32 bit int
    for (int i=0; i < NROWS; i++) {
        fwrite(&data[i*NCOLS], sizeof(float), NCOLS, stream); // write width as 32 bit int
    }
    //fwrite(data, sizeof(uint8_t), len, stream); // write width as 32 bit int
    fclose(stream); 

    free(shape);
    free(padding);
    free(npy_hdr);
}