import copy
import numpy as np
from pathlib import Path

from .bil import Bil
from .bsq import Bsq
from .tif import Tif
from ._ext import *

class Raster(Bil, Bsq, Tif):

    def __init__(self, data=None, **kwargs):
        
        # assign data
        self.raster = data

        # assign raster metadata
        if data is not None:
            self.NROWS = kwargs.get('NROWS', self.raster.shape[0])
            self.NCOLS = kwargs.get('NCOLS', self.raster.shape[0])
            self.NBANDS = kwargs.get('NBANDS', self.raster.ndim - 1)
            self.NBITS = kwargs.get('NBITS', self.raster.itemsize * 8)

            if issubclass(self.raster.dtype.type, np.floating):
                inferrted_PT = 'FLOAT'
            elif issubclass(self.raster.dtype.type, np.uint8):
                inferrted_PT = 'UNSIGNEDINT'
            self.PIXELTYPE = kwargs.get('PIXELTYPE', inferrted_PT)   
            self.NPTYPE = self.raster.dtype  
        else:
            self.NROWS = None
            self.NCOLS = None
            self.NBANDS = None
            self.NBITS = None
            self.PIXELTYPE = None
            self.NPTYPE = None

        # assign geo metadata
        self.ULXMAP = kwargs.get('ULXMAP', 0)
        self.ULYMAP = kwargs.get('ULYMAP', 0)
        self.XDIM = kwargs.get('XDIM', 1)
        self.YDIM = kwargs.get('YDIM', 1)
        self.NODATA = kwargs.get('NODATA', None)
        self.NODATA = kwargs.get('NODATA', None)

        # assign number of threads
        self.NUMT = kwargs.get('NUMT', 1)
    
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result
    
    def copy(self):
        return self.__copy__()

    def deepcopy(self):
        return self.__deepcopy__(memo={})

    def read(self, fname, hname=None):
        """read a raster file and construct Raster object

        Args:
            fname (filename): path to the filename containing data
            hname (filename, optional): path to any required header file
        """

        # make sure we are using pathlib objects
        if isinstance(fname, str):
            fname = Path(fname)

        if isinstance(hname, str):
            hname = Path(hname)

        if fname.suffix == '.bil':
            # load bil
            if hname is None:
                hname = fname.with_suffix('.hdr')
            self.read_hdr(hname)
            self.raster = self.read_bil(fname)
        elif fname.suffix == '.bsq':
            # load bsq
            if hname is None:
                hname = fname.with_suffix('.hdr')
            self.read_bsq_hdr(hname)
            self.raster = self.read_bsq(fname)
        elif fname.suffix == '.tif':
            # load tif
            if hname is None:
                hname = fname.with_suffix('.tfw')
            self.read_tfw(hname)
            self.raster = self.read_tif(fname)
        else:
            raise ValueError(f'filetype {fname.suffix} not supported for read!')

    def write(self, fname, hname=None):
        """write a raster file from the Raster object

        Args:
            fname (filename): path to the filename containing data
            hname (filename, optional): path to any required header file
        """

        if np.isnan(self.raster).any():
            raise ValueError('Remove NaNs before writing file!')

        # make sure we are using pathlib objects
        if isinstance(fname, str):
            fname = Path(fname)
        if isinstance(hname, str):
            hname = Path(hname)

        if fname.suffix == '.bil':
            if hname is None:
                hname = fname.with_suffix('.hdr')
            self.raster = self.raster.astype(self.NPTYPE)
            self.write_bil(self.raster, fname)
            self.write_hdr(hname)
        elif fname.suffix == '.bsq':
            if hname is None:
                hname = fname.with_suffix('.hdr')
            self.raster = self.raster.astype(self.NPTYPE)
            self.write_bsq(self.raster, fname)
            self.write_bsq_hdr(hname)
        elif fname.suffix == '.tif':
            if hname is None:
                hname = fname.with_suffix('.tfw')
            self.write_tif(self.raster, fname)
            self.write_tfw(hname)
        else:
            raise ValueError(f'file type {fname.suffix} not supported for write!')

    def noData2Nan(self):
        # convert no Data to NAN
        self.raster[self.raster==self.NODATA] = np.nan

    def nan2NoData(self):
        # convert NAN to no Data
        self.raster[np.isnan(self.raster)] = self.NODATA

    def imageToReal(self, x, y):
        X = self.ULXMAP + (self.XDIM * x)
        Y = self.ULYMAP -(self.YDIM * y)
        return X, Y

    def realToImage(self, X, Y):
        x = (X - self.ULXMAP) / self.XDIM
        y = (self.ULYMAP - Y) / self.YDIM
        return x, y

    def slope(self, numt=None):
        """Compute the slope of a DEM

        Args:
            numt (int, optional): number of threads used to compute hillshade

        Returns:
            2D array: 2D grid of slope values (radians)
        """
        if len(self.raster.shape) > 2:
            raise ValueError(f'Raster contains more than 2 dimensions')
        if self.raster.dtype != np.float32:
            raise ValueError(f'Raster must be type np.float32')
        if numt is None:
            return slope_mp(self.raster, self.XDIM, self.NUMT)
        elif isinstance(numt, int):
            return slope_mp(self.raster, self.XDIM, numt)

    def hillshade(self, azimuth=330, altitude=30, numt=None):
        """Compute a hillshade of a DEM

        Args:
            azimuth (int, optional): Azimuth used to compute hillshade
            altitude (int, optional): Altitude used to compute hillshade
            numt (int, optional): number of threads used to compute hillshade

        Returns:
            2D array: 2D grid of hillshade values
        """
        if len(self.raster.shape) > 2:
            raise ValueError(f'Raster contains more than 2 dimensions')
        if self.raster.dtype != np.float32:
            raise ValueError(f'Raster must be type np.float32')
        if numt is None:
            return hillshade_mp_faster(self.raster, azimuth, altitude, self.XDIM, self.NUMT)
        elif isinstance(numt, int):
            return hillshade_mp_faster(self.raster, azimuth, altitude, self.XDIM, numt)

    def aspect(self, numt=None):
        """Compute the aspect of a DEM

        Args:
            numt (int, optional): number of threads used to compute hillshade

        Returns:
            2D array: 2D grid of aspect values (radians)
        """
        if len(self.raster.shape) > 2:
            raise ValueError(f'Raster contains more than 2 dimensions')
        if self.raster.dtype != np.float32:
            raise ValueError(f'Raster must be type np.float32')
        if numt is None:
            return aspect_mp(self.raster, self.XDIM, self.NUMT)
        elif isinstance(numt, int):
            return aspect_mp(self.raster, self.XDIM, numt)

    def avg_aspect(self):
        # find and apply Z factor of DEM
        aspect = self.aspect()
        return np.arctan2(np.sin(aspect).sum(), np.cos(aspect).sum())  % (2*np.pi)

    def crop(self, xmin, xmax, ymin, ymax):
        """crop raster to provided img coords"""
        self.raster = self.raster[ymin:ymax,xmin:xmax]
        self.NROWS = self.raster.shape[0]
        self.NCOLS = self.raster.shape[1]
        self.ULXMAP, self.ULYMAP = self.imageToReal(xmin, ymin)
        self.ULXMAP, self.ULYMAP = np.round(self.ULXMAP, 3), np.round(self.ULYMAP, 3)
        return 0

    def geo_crop(self, Xmin, Xmax, Ymin, Ymax):
        """crop to geographical extents"""

        # get extents in img coords
        xmin, ymin = self.realToImage(Xmin, Ymax)
        xmax, ymax = self.realToImage(Xmax, Ymin)

        # round to nearest int
        xmin, ymin = int(round(xmin)), int(round(ymin))
        xmax, ymax = int(round(xmax)), int(round(ymax))

        # crop raster
        self.crop(xmin, xmax, ymin, ymax)
        return 0

    def resize(self, new_rows, new_cols, METHOD='linear'):
        from scipy.interpolate import RegularGridInterpolator as RGI

        rowScale = self.NROWS / new_rows
        colScale = self.NCOLS / new_cols
        interp_func = RGI((np.arange(self.NCOLS), np.arange(self.NROWS)), self.raster.T, method=METHOD)

        xi = np.linspace(0, self.NCOLS - 1, new_cols)
        yi = np.linspace(0, self.NROWS - 1, new_rows)
        X, Y = np.meshgrid(xi, yi)
        new_grid = np.stack(np.meshgrid(xi, yi), axis=-1)

        # perform interpolating and check new shape is correct
        self.raster = interp_func(new_grid)
        assert (self.raster.shape[0] == new_rows) and (self.raster.shape[1] == new_cols)

        # update raster meta-data
        self.NROWS = self.raster.shape[0]
        self.NCOLS = self.raster.shape[1]
        self.ULXMAP -= (self.XDIM / 2)
        self.ULYMAP += (self.YDIM / 2)
        self.XDIM *= colScale
        self.YDIM *= rowScale
        self.ULXMAP += (self.XDIM / 2)
        self.ULYMAP -= (self.YDIM / 2)
        return 0

    def fill_nans(self, num_ctrl=50, s=1.0):
        """fills nans in 2D numpy grid using scipy Rbf TPS

        Args:
            num_ctrl (int, optional): number of ctrl pts to use to create spline. Defaults to 50.
            s (float, optional): smoothing factor of tps. Defaults to 1.0.

        Returns:
            2D array: 2D grid with nans filled
        """
        from scipy.interpolate import RBFInterpolator

        # TODO: replace with more robust openMP implementation

        # Setup
        rows = self.raster.shape[0]
        cols = self.raster.shape[1]
        pts = self.raster
        x, y = np.arange(cols), np.arange(rows)
        X, Y = np.meshgrid(x, y)         

        # get initial ctrl pts by removing nans
        Xc = X[~np.isnan(pts)]
        Yc = Y[~np.isnan(pts)]
        Zc = pts[~np.isnan(pts)]

        # subsample ctrl evenly 
        idx = np.round(np.linspace(0, len(Zc) - 1, num_ctrl)).astype(int)
        Xc = Xc[idx]
        Yc = Yc[idx]
        Zc = Zc[idx]
        Hc = np.hstack((Xc.reshape(Xc.size, 1), Yc.reshape(Yc.size, 1)))

        # Generate Spline surface
        spline = RBFInterpolator(Hc, Zc, smoothing=s)
        
        # fill nans from spline
        Xi = X[np.isnan(pts)]
        Yi = Y[np.isnan(pts)]
        xi = np.hstack((Xi.reshape(Xi.size, 1), Yi.reshape(Yi.size, 1)))
        Zi = spline(xi)
        pts[np.isnan(pts)] = Zi
        return pts

    def set_num_threads(self, numt=1):
        """Set the # of threads to use for openMP operations

        Args:
            numt (int, optional): number of openMP threads
        """ 
        self.NUMT = numt

def read(fname, hname=None):
    """read a raster file and construct Raster object

    Args:
        fname (filename): path to the filename containing data
        hname (filename, optional): path to any required header file

    Returns:
        Raster: object containing raster data
    """
    raster = Raster()
    raster.read(fname, hname)
    return raster

def slope(dem, cell_size, numt=1):
    """Compute the slope of a DEM

    Args:
        dem (np.array): 2D array representing the dem
        cell_size (float): cellsize of the dem
        numt (int, optional): number of threads used to compute hillshade

    Returns:
        2D array: 2D grid of slope values (radians)
    """
    if len(dem.shape) > 2:
        raise ValueError(f'Raster contains more than 2 dimensions')
    if dem.dtype != np.float32:
        raise ValueError(f'Raster must be type np.float32')

    return slope_mp(dem, cell_size, numt)

def hillshade(dem, cell_size, azimuth=330, altitude=30, numt=None):
    """Compute a hillshade of a DEM

    Args:
        dem (np.array): 2D array representing the dem
        cell_size (float): cellsize of the dem
        azimuth (int, optional): Azimuth used to compute hillshade
        altitude (int, optional): Altitude used to compute hillshade
        numt (int, optional): number of threads used to compute hillshade

    Returns:
        2D array: 2D grid of hillshade values
    """
    if len(dem.shape) > 2:
        raise ValueError(f'Raster contains more than 2 dimensions')
    if dem.dtype != np.float32:
        raise ValueError(f'Raster must be type np.float32')
    return hillshade_mp_faster(dem, azimuth, altitude, cell_size, numt)

def aspect(dem, cell_size, numt=None):
    """Compute the aspect of a DEM

    Args:
        numt (int, optional): number of threads used to compute hillshade

    Returns:
        2D array: 2D grid of aspect values (radians)
    """
    if len(dem.shape) > 2:
        raise ValueError(f'Raster contains more than 2 dimensions')
    if dem.dtype != np.float32:
        raise ValueError(f'Raster must be type np.float32')
    
    return aspect_mp(dem, cell_size, numt)

def convolve2d(a, f):
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape = s, strides = a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)

def inpaint_nans(im):
    ipn_kernel = np.array([[1,1,1],[1,0,1],[1,1,1]]) # kernel for inpaint_nans
    nans = np.isnan(im)
    while np.sum(nans)>0:
        im[nans] = 0
        vNeighbors = np.pad(convolve2d((nans==False), ipn_kernel), 1, mode='edge')
        im2 = np.pad(convolve2d(im, ipn_kernel), 1, mode='edge')
        im2[vNeighbors>0] = im2[vNeighbors>0]/vNeighbors[vNeighbors>0]
        im2[vNeighbors==0] = np.nan
        im2[(nans==False)] = im[(nans==False)]
        im = im2
        nans = np.isnan(im)
    return im
