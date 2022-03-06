import copy
import numpy as np
from scipy.interpolate import RBFInterpolator

class Flo:

    def read_fow(self, hname):
        """read the flo world file
        Args:
            hname (Path object): Path object to flo world file

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

    def write_fow(self, hname):
        with open(hname, 'w', encoding = 'utf-8') as f:
            f.write(f'{self.XDIM:.6f}\n')
            f.write(f'{self.YROT:.6f}\n')
            f.write(f'{self.XROT:.6f}\n')
            f.write(f'{-self.YDIM:.6f}\n')
            f.write(f'{self.ULXMAP:.6f}\n')
            f.write(f'{self.ULYMAP:.6f}\n')
        return 0

    def read_flo(self, fname):
        with open(fname, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)[0]
            if not magic == 202021.25:
                raise IOError(f'file: {fname} is invalid flo file')
            self.NCOLS = np.fromfile(f, np.int32, count=1)[0]
            self.NROWS = np.fromfile(f, np.int32, count=1)[0]
            array = np.fromfile(f, np.float32, count = self.NBANDS * self.NROWS * self.NCOLS)
        return array.reshape(self.NROWS, self.NCOLS, self.NBANDS)

    def write_flo(self, fname):
        with open(fname, 'wb') as f:
            magic = np.array([202021.25], dtype=np.float32)
            magic.tofile(f)
            NROWS, NCOLS = self.raster.shape[:2]
            np.int32(NCOLS).tofile(f)
            np.int32(NROWS).tofile(f)
            self.raster.flatten().tofile(f)
        return 0

    def __init__(self, fname, data=None):
        #TODO: Implement geo flo
        self.NBANDS = 2
        if fname is not None:
            self.raster = self.read_flo(fname)
        elif data is not None:
            self.raster = data

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

    def crop(self, xmin, xmax, ymin, ymax):
        """crop flo raster to provided img coords"""
        self.raster = self.raster[ymin:ymax,xmin:xmax, :]
        self.NROWS = self.raster.shape[0]
        self.NCOLS = self.raster.shape[1]
        return 0

    def get_u(self, view=True):
        if view == True:
            return self.raster[:,:,::2].reshape(self.NROWS, self.NCOLS)
        elif view == False:
            return self.raster[:,:,::2].reshape(self.NROWS, self.NCOLS).copy()
        else:
            raise ValueError(f'Invalid view param provided: {view}')
    
    def get_v(self, view=True):
        if view == True:
            return self.raster[:,:,1::2].reshape(self.NROWS, self.NCOLS)
        elif view == False:
            return self.raster[:,:,1::2].reshape(self.NROWS, self.NCOLS).copy()
        else:
            raise ValueError(f'Invalid view param provided: {view}')

    def flo_2_real(self, XDIM, YDIM):
        """Convert flo from pixels to real world units

        Args:
            XDIM (float): units of cell in X
            YDIM (float): units of cell in Y
        """
        self.raster[:,:,0] = self.raster[:,:,0] * XDIM
        self.raster[:,:,1] = -self.raster[:,:,1] * YDIM

    def real_2_flo(self, XDIM, YDIM):
        """Convert flo from real world units to pixels

        Args:
            XDIM (float): units of cell in X
            YDIM (float): units of cell in Y
        """
        self.raster[:,:,0] = self.raster[:,:,0] / XDIM
        self.raster[:,:,1] = -self.raster[:,:,1] / YDIM

    def fill_nans(self, num_ctrl=50, s=1.0):
        rows = self.raster.shape[0]
        cols = self.raster.shape[1]
        pts = self.raster
        x, y, k = np.arange(cols), np.arange(rows), np.array((0,1))
        X, Y, K = np.meshgrid(x, y, k)         

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

    def local_smooth(self, umin, umax, vmin, vmax, num_ctrl=None):
        """Smooth outliers using spline

        Args:
            thres (tuple): outlier threshold (u & v)
            num_ctrl (int): num ctrl pts for spline
        """

        # ID outliers
        u_outliers = (self.raster[:,:,0] < umin) | (self.raster[:,:,0] > umax)
        v_outliers = (self.raster[:,:,1] < vmin) | (self.raster[:,:,1] > vmax)

        self.raster[:,:,0][u_outliers] = np.nan
        self.raster[:,:,1][v_outliers] = np.nan

        print(np.count_nonzero(np.isnan(self.raster)), ' nans')

        if num_ctrl is None:
            self.raster = self.fill_nans()
        else:
            self.raster = self.fill_nans(num_ctrl)

    def flo_to_quiver(self, ax, step=50, margin=0, **kwargs):
        """Add flo to matplotlib quiver plot

        Args:
            ax: matplotlib axis
            steps: space (px) between each arrow in grid
            margin: margin (px) of enclosing region without arrows
            kwargs: quiver kwargs (default: angles="xy", scale_units="xy")
        """

        nx = int((self.raster.shape[1] - 2 * margin) / step)
        ny = int((self.raster.shape[0] - 2 * margin) / step)
        x = np.linspace(margin, self.raster.shape[1] - margin - 1, nx, dtype=np.int64)
        y = np.linspace(margin, self.raster.shape[0] - margin - 1, ny, dtype=np.int64)
        
        flow = self.raster[np.ix_(y, x)]
        u = flow[:, :, 0]
        v = flow[:, :, 1]

        kwargs = {**dict(angles="xy", scale_units="xy"), **kwargs}

        ax.quiver(x, y, u, v, **kwargs)
