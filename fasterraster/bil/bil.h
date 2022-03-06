#pragma once

void write_hdr(const char* filename, const int NROWS, const int NCOLS, const double ULXMAP, const double ULYMAP, const float XDIM, const float YDIM, const float NODATA);

void write_bil(const float* dem, const char* filename, const int NROWS, const int NCOLS);

void read_bil(float* data, const char* fname, const int NROWS, const int NCOLS);