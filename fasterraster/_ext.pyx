# distutils: language = c
# distutils: sources = bil/bil.c, flo/flo.c, npy/npy.c, operations/operations.c

import numpy as np
cimport cython
cimport numpy as np
from cython.parallel cimport prange
from libc.stdint cimport uint8_t

##############################################################
##################### Bil Operations #########################
##############################################################
cdef extern from "bil/bil.h":
    void write_hdr(const char* filename, const int NROWS, const int NCOLS, const float ULXMAP, const float ULYMAP, const float XDIM, const float YDIM, const float NODATA)
    void write_bil(const float* dem, const char* filename, const int NROWS, const int NCOLS)
    void read_bil(float* data, const char* fname, const int NROWS, const int NCOLS)

def read_bil_float32(fname, const int rows, const int cols):
    cdef bytes py_bytes = fname.encode()
    cdef char* fname_ptr = py_bytes

    data = np.empty((rows, cols), dtype=np.float32, order='C')
    cdef float[:,::1] data_view = data
    read_bil(&data_view[0,0], &fname_ptr[0], rows, cols)
    return data

def write_bil_float32(float[:,::1] dem, double ulxmap, double ulymap, float xdim, float ydim, float nodata, char* fname, char* hname):
    cdef int rows = int(dem.shape[0])
    cdef int cols = int(dem.shape[1])
    
    write_hdr(&hname[0], rows, cols, ulxmap, ulymap, xdim, ydim, nodata)
    write_bil(&dem[0,0], &fname[0], rows, cols)

##############################################################
##################### Flo Operations #########################
##############################################################
cdef extern from "flo/flo.h" nogil:
    void construct_flo(float* flo, const float* u, const float* v, const int SIZE)
    void get_u_v(float* u, float* v, const float* flo, const int SIZE, const int NUMT)
    void write_flo(const float* flo, const char* filename, const int NROWS, const int NCOLS)
    void write_flo_MP(const float* u, const float* v, const char* fbase, const int NUMF, const int NROWS, const int NCOLS, const int START_IDX, const int NUMT)
    void read_flo(float* flo, const char* filename, const int NROWS, const int NCOLS)
    void multiply_flo_scalers(float* cpy_flo, const float* src_flo, const int ITERS, const int NROWS, const int NCOLS, const int NUMT)
    void multiply_flo_mask_arr(float* arr1, const uint8_t* arr2, const int SIZE, const int NUMT)
    void multiply_flo_and_save(const float* u, const float* v, const char* filename, const int iters, const int NROWS, const int NCOLS, const int NUMT)

def read_flo_float32(fname, const int rows, const int cols):
    cdef bytes py_bytes = fname.encode()
    cdef char* fname_ptr = py_bytes
    
    data = np.empty((rows, cols, 2), dtype=np.float32, order='C')
    cdef float[:,:,::1] data_view = data
    read_flo(&data_view[0,0,0], &fname_ptr[0], rows, cols)

    return data

def write_flo_float32(const float[:,::1] u, const float[:,::1] v, char* fname):
    cdef int rows = int(u.shape[0])
    cdef int cols = int(u.shape[1])

    flo_np = np.empty((rows, cols, 2), dtype=np.float32, order='C')
    cdef float[:,:,::1] flo = flo_np

    construct_flo(&flo[0,0,0], &u[0,0], &v[0,0], rows*cols)
    write_flo(&flo[0,0,0], &fname[0], rows, cols)

def flo_to_u_v(const float[:,:,::1] flo, const int numt=1):
    cdef int rows = int(flo.shape[0])
    cdef int cols = int(flo.shape[1])
    u = np.empty((rows, cols), dtype=np.float32, order='C')
    v = np.empty((rows, cols), dtype=np.float32, order='C')
    cdef float[:,::1] u_view = u
    cdef float[:,::1] v_view = v
    get_u_v(&u_view[0,0], &v_view[0,0], &flo[0,0,0], rows*cols, numt)
    return u, v

def multiplyFloScalers(float[:,:,:,::1] cpy_flo, float[:,:,::1] src_flo, const int numt=1):
    cdef int iters = int(cpy_flo.shape[0])
    cdef int rows = int(cpy_flo.shape[1])
    cdef int cols = int(cpy_flo.shape[2])

    multiply_flo_scalers(&cpy_flo[0,0,0,0], &src_flo[0,0,0], iters, rows, cols, numt)

def multiplyFloMask(float[:,:,::1] flo, const uint8_t[:,::1] mask, const int numt=1):
    cdef int size = mask.size

    multiply_flo_mask_arr(&flo[0,0,0], &mask[0,0], size, numt)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def parallel_write_flo(const float[:,:,::1] u, const float[:,:,::1] v, fbase, const int start_idx, const int numt=1):
    
    cdef int numF = int(u.shape[0])
    cdef int rows = int(u.shape[1])
    cdef int cols = int(u.shape[2])

    cdef bytes py_bytes = fbase.encode()
    cdef char* fbase_ptr = py_bytes

    # save flo files in parallel
    write_flo_MP(&u[0,0,0], &v[0,0,0], &fbase_ptr[0], numF, rows, cols, start_idx, numt)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def save_sim_flo(const float[:,::1] u, const float[:,::1] v, const int num_iter, fname, const int numt=1):
    cdef int rows = int(u.shape[0])
    cdef int cols = int(u.shape[1])

    # save flo files in bulk from simulation output
    cdef bytes py_bytes = fname.encode()
    cdef char* fname_ptr = py_bytes
    multiply_flo_and_save(&u[0,0], &v[0,0], &fname_ptr[0], num_iter, rows, cols, numt)

##############################################################
##################### Npy Operations #########################
##############################################################
cdef extern from "npy/npy.h" nogil:
    void write_u8bit_npy(uint8_t* data, const char* filename, const int NROWS, const int NCOLS)
    void write_float32_npy(float* data, const char* filename, const int NROWS, const int NCOLS)

def write_npy_uint8(uint8_t[:, ::1] data, char* fname):
    cdef int rows = int(data.shape[0])
    cdef int cols = int(data.shape[1])
    write_u8bit_npy(&data[0,0], &fname[0], rows, cols)

def write_npy_float32(float[:, ::1] data, char* fname):
    cdef int rows = int(data.shape[0])
    cdef int cols = int(data.shape[1])
    write_float32_npy(&data[0,0], &fname[0], rows, cols)

##############################################################
################## Raster Operations #########################
##############################################################
cdef extern from "operations/operations.h":
    void HILLSHADE_MP(const float *dem, uint8_t *hs, const int AZIMUTH, const int ALITITUDE, const float CELL_SIZE, const int NROWS, const int NCOLS, const int NUMT)
    void HILLSHADE_MP_FASTER(const float *dem, uint8_t *hs, const int AZIMUTH, const int ALITITUDE, const float CELL_SIZE, const int NROWS, const int NCOLS, const int NUMT)
    void SLOPE_MP(const float *dem, float *slope, const float CELL_SIZE, const int NROWS, const int NCOLS, const int NUMT)
    void HILLSHADE_SLOPE_MP(const float *dem, uint8_t *hs, float *slope, const int AZIMUTH, const int ALITITUDE, const float CELL_SIZE, const int NROWS, const int NCOLS, const int NUMT)
    void ASPECT_MP(const float *dem, float *aspect, const float CELL_SIZE, const int NROWS, const int NCOLS, const int NUMT)
    void STATIC_CENTER_CROP_MP(float* crop, const float* full, const int crop_NROWS, const int crop_NCOLS, const int full_NROWS, const int full_NCOLS, const int NUMT)
    void STATIC_RANDOM_CROP_MP(float* crop, const float* full, const int crop_NROWS, const int crop_NCOLS, const int full_NROWS, const int full_NCOLS, const int NUMT)
    void MULTIPY_SCALER_MP(float* arr, const float SCALER, const int SIZE, const int NUMT)
    void MULTIPY_SCALER_SSE(float* arr, const float SCALER, const int SIZE, const int NUMT)

def hillshade_mp(float[:,::1] dem, const int azimuth, const int altitude, const float cell_size, const int numt=1):
    cdef int rows = int(dem.shape[0])
    cdef int cols = int(dem.shape[1])

    hs_numpy = np.empty((rows, cols), dtype=np.uint8, order='C')
    cdef uint8_t [:,::1] hs = hs_numpy
    HILLSHADE_MP(&dem[0,0], &hs[0,0], azimuth, altitude, cell_size, rows, cols, numt)
    return hs_numpy

def hillshade_mp_faster(const float[:,::1] dem, const int azimuth, const int altitude, const float cell_size, const int numt=1):
    cdef int rows = int(dem.shape[0])
    cdef int cols = int(dem.shape[1])

    hs_numpy = np.empty((rows, cols), dtype=np.uint8, order='C')
    cdef uint8_t [:,::1] hs = hs_numpy
    HILLSHADE_MP_FASTER(&dem[0,0], &hs[0,0], azimuth, altitude, cell_size, rows, cols, numt)
    return hs_numpy

def slope_mp(const float[:,::1] dem, const float cell_size, const int numt=1):
    cdef int rows = int(dem.shape[0])
    cdef int cols = int(dem.shape[1])
    slope_numpy = np.empty((rows, cols), dtype=np.float32, order='C')
    cdef float [:,::1] slope = slope_numpy
    SLOPE_MP(&dem[0,0], &slope[0,0], cell_size, rows, cols, numt)
    return slope_numpy

def hillshade_slope_mp(const float[:,::1] dem, const int azimuth, const int altitude, const float cell_size, const int numt=1):
    cdef int rows = int(dem.shape[0])
    cdef int cols = int(dem.shape[1])

    hs_numpy = np.empty((rows, cols), dtype=np.uint8, order='C')
    cdef uint8_t [:,::1] hs = hs_numpy
    
    slope_numpy = np.empty((rows, cols), dtype=np.float32, order='C')
    cdef float [:,::1] slope = slope_numpy
    
    HILLSHADE_SLOPE_MP(&dem[0,0], &hs[0,0], &slope[0,0], azimuth, altitude, cell_size, rows, cols, numt)
    return hs_numpy, slope_numpy

def aspect_mp(const float [:,::1] dem, const float cell_size, const int numt=1):
    cdef int rows = int(dem.shape[0])
    cdef int cols = int(dem.shape[1])
    aspect_numpy = np.empty((rows, cols), dtype=np.float32, order='C')
    cdef float [:,::1] aspect = aspect_numpy
    ASPECT_MP(&dem[0,0], &aspect[0,0], cell_size, rows, cols, numt)
    return aspect_numpy

def static_center_crop_mp(const float[:,::1] raster, crop_size, const int numt=1):

    cdef int full_rows = int(raster.shape[0])
    cdef int full_cols = int(raster.shape[1])
    cdef int crop_rows = int(crop_size[0])
    cdef int crop_cols = int(crop_size[1])

    crop = np.empty(crop_size, dtype = np.float32)
    cdef float[:,::1] crop_view = crop

    STATIC_CENTER_CROP_MP(&crop_view[0,0], &raster[0,0], crop_rows, crop_cols, full_rows, full_cols, numt)
    return crop

def static_random_crop_mp(const float[:,::1] raster, crop_size, const int numt=1):

    cdef int full_rows = int(raster.shape[0])
    cdef int full_cols = int(raster.shape[1])
    cdef int crop_rows = int(crop_size[0])
    cdef int crop_cols = int(crop_size[1])

    crop = np.empty(crop_size, dtype = np.float32)
    cdef float[:,::1] crop_view = crop

    STATIC_RANDOM_CROP_MP(&crop_view[0,0], &raster[0,0], crop_rows, crop_cols, full_rows, full_cols, numt)
    return crop

def multiply_scaler_mp(float[::1] arr, const float scaler, const int numt=1):
    cdef int size = arr.size
    MULTIPY_SCALER_MP(&arr[0], scaler, size, numt)

def multiply_scaler_SSE(float[::1] arr, const float scaler, const int numt=1):
    cdef int size = arr.size
    MULTIPY_SCALER_SSE(&arr[0], scaler, size, numt)
