import numpy as np
import tifffile
"""Wrapper class for reading and writing tifs using tifffile"""

# TODO: Update to remove tifffile as dependancy

class Tif:

    def read_tfw(self, hname):
        """read the tif header file
        Args:
            hname (Path object): Path object to header file

        Returns:
            int: 0 if read successfully
        """
        with open(hname, 'r', encoding = 'utf-8') as f:
            self.XDIM = np.float32(next(f))
            self.YROT = float(next(f))
            self.XROT = float(next(f))
            self.YDIM = np.float32(next(f))
            self.ULXMAP = float(next(f))
            self.ULYMAP = float(next(f))
        self.YDIM = np.abs(self.YDIM)
        return 0

    def write_tfw(self, hname):
        with open(hname, 'w', encoding = 'utf-8') as f:
            f.write(f'{self.XDIM:.6f}\n')
            f.write(f'{self.YROT:.6f}\n')
            f.write(f'{self.XROT:.6f}\n')
            f.write(f'{-self.YDIM:.6f}\n')
            f.write(f'{self.ULXMAP:.6f}\n')
            f.write(f'{self.ULYMAP:.6f}\n')
        return 0

    def read_tif(self, fname):
        array = tifffile.imread(fname)
        try:
            self.NROWS, self.NCOLS, self.NBANDS = array
        except ValueError:
            self.NROWS, self.NCOLS, *_ = array
            self.NBANDS = 1
        return array

    def write_tif(self, data, fname):
        tifffile.imwrite(fname, data)
        return 0

    def __init__(self, fname, hname):
        self.read_tfw(hname)
        self.raster = self.read_tif(fname)
