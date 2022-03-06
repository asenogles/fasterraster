import warnings
import numpy as np

class Bsq:

    def read_bsq_hdr(self, hname):
        """read the bsq header file
        Args:
            hname (Path object): Path object to header file

        Returns:
            int: 0 if read successfully
        """
        with open(hname, 'r', encoding = 'utf-8') as f:
            for i, line in enumerate(f):
                words = line.split()
                if len(words) != 2:
                    print(f'line {i+1} of hdr not read, too many values:')
                    print(f'line {i+1}: {" ".join(map(str, words))}')
                    continue

                fieldName, value = words
                if fieldName == 'BYTEORDER':
                    self.BYTEORDER = value
                elif fieldName == 'LAYOUT':
                    self.LAYOUT = value
                elif fieldName == 'NROWS':
                    self.NROWS = int(value)
                elif fieldName == 'NCOLS':
                    self.NCOLS = int(value)
                elif fieldName == 'NBANDS':
                    self.NBANDS = int(value)
                elif fieldName == 'NBITS':
                    self.NBITS = int(value)
                    self.NBYTES = int(self.NBITS // 8)
                elif fieldName == 'PIXELTYPE':
                    self.PIXELTYPE = str(value)
                elif fieldName == 'ULXMAP':
                    self.ULXMAP = float(value)
                elif fieldName == 'ULYMAP':
                    self.ULYMAP = float(value)
                elif fieldName == 'XDIM':
                    self.XDIM = np.float32(value)
                elif fieldName == 'YDIM':
                    self.YDIM = np.float32(value)
                elif fieldName == 'NODATA':
                    self.NODATA = np.float32(value)
                else:
                    continue
            # assign type
            if hasattr(self, 'PIXELTYPE'):
                if self.NBYTES == 1 and self.PIXELTYPE == 'UNSIGNEDINT':
                    self.TYPE = 'B'
                    self.NPTYPE = np.uint8
                elif self.NBYTES == 1 and self.PIXELTYPE == 'SIGNEDINT':
                    self.TYPE = 'b'
                    self.NPTYPE = np.int8
                elif self.NBYTES == 4 and self.PIXELTYPE == 'UNSIGNEDINT':
                    self.TYPE = 'I'
                    self.NPTYPE = np.uint32
                elif self.NBYTES == 4 and self.PIXELTYPE == 'SIGNEDINT':
                    self.TYPE = 'i'
                    self.NPTYPE = np.int32
                elif self.NBYTES == 4 and self.PIXELTYPE == 'FLOAT':
                    self.TYPE = 'f'
                    self.NPTYPE = np.float32
                else:
                    # not found, pretend doesn't exist
                    del self.PIXELTYPE

            if not hasattr(self, 'PIXELTYPE'):
                # issue warning
                warnings.warn('PIXELTYPE not defined in header file, assumming PIXELTPYE based on NBITS field')
                if self.NBYTES == 1:
                    self.TYPE = 'c'
                    self.NPTYPE = np.int8
                    print('assigned type char')
                elif self.NBYTES == 4:
                    self.TYPE = 'f'
                    self.NPTYPE = np.float32
                    print('assigned type 32 bit float')

        return 0

    def write_bsq_hdr(self, hname):
        # get metadata
        NROWS = self.raster.shape[0]
        NCOLS = self.raster.shape[1]
        NBANDS = self.raster.ndim-1
        NBITS = self.raster.itemsize * 8
        BANDROWBYTES = self.raster.itemsize * NROWS
        TOTALROWBYTES = self.raster.itemsize * NROWS * NBANDS
        if issubclass(self.raster.dtype.type, np.floating):
            PIXELTYPE = 'FLOAT'
        elif issubclass(self.raster.dtype.type, np.uint8):
            PIXELTYPE = 'UNSIGNEDINT'
        elif issubclass(self.raster.dtype.type, np.uint32):
            PIXELTYPE = 'UNSIGNEDINT'
        elif issubclass(self.raster.dtype.type, np.integer):
            PIXELTYPE = 'INT'

        # write to hdr file
        with open(hname, 'w', encoding = 'utf-8') as f:
            f.write(f'BYTEORDER      I\n')
            f.write(f'LAYOUT         BSQ\n')
            f.write(f'NROWS          {NROWS}\n')
            f.write(f'NCOLS          {NCOLS}\n')
            f.write(f'NBANDS         {NBANDS}\n')
            f.write(f'NBITS          {NBITS:d}\n')
            f.write(f'BANDROWBYTES   {BANDROWBYTES:d}\n')
            f.write(f'TOTALROWBYTES  {TOTALROWBYTES:d}\n')
            f.write(f'PIXELTYPE      {PIXELTYPE}\n')
            f.write(f'ULXMAP         {self.ULXMAP:.3f}\n')
            f.write(f'ULYMAP         {self.ULYMAP:.3f}\n')
            f.write(f'XDIM           {self.XDIM:.3f}\n')
            f.write(f'YDIM           {self.YDIM:.3f}\n')
            f.write(f'NODATA         {self.NODATA:.2f}\n')
        return 0

    def read_bsq(self, fname):
        with open(fname, 'rb') as f:
            buffer = f.read()
            array = np.frombuffer(buffer, self.NPTYPE)
        return array.reshape(self.NBANDS, self.NROWS, self.NCOLS)

    def write_bsq(self, data, fname):
        
        b = data.tobytes()
        with open(fname, 'wb') as f:
            f.write(b)
        return 0

    def __init__(self, fname, hname):
        self.read_bsq_hdr(hname)
        self.raster = self.read_bsq(fname)
