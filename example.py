import timeit
import numpy as np
import fasterraster as fr
from pathlib import Path

NTESTS = 10

# Load a .bil file containing a DEM
fname = Path('./test_data/dem.bil')
dem = fr.read(fname)

# regular python implementation of hillshade function
# from https://www.neonscience.org/resources/learning-hub/tutorials/create-hillshade-py
def py_hillshade(dem, cell_size, azimuth=330, altitude=30):
    azimuth = 360.0 - azimuth

    dem = dem / cell_size
    x, y = np.gradient(dem)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth*np.pi/180.
    altituderad = altitude*np.pi/180.
 
    shaded = np.sin(altituderad)*np.sin(slope)
    + np.cos(altituderad)*np.cos(slope) * np.cos(
    (azimuthrad - np.pi/2.) - aspect)
    
    return 255*(shaded + 1)/2

# Time hillshade computation using regular python
time = timeit.timeit(lambda: py_hillshade(dem.raster, dem.XDIM), number=NTESTS)
print(f'python hillshade averaged {time/NTESTS:.3f} seconds')

# Time hillshade computation using open-MP for numt-threads
num_threads = [1,2,4,8]
for numt in num_threads:
    time = timeit.timeit(lambda: fr.hillshade_mp_faster(dem.raster, 330, 30, dem.XDIM, numt), number=10)
    print(f'hillshade averaged {time/NTESTS:.3f} seconds for {numt} threads')