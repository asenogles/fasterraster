# Faster-Raster

[![pypi](https://img.shields.io/pypi/v/fasterraster.svg)](https://pypi.python.org/pypi/fasterraster)
[![image](https://img.shields.io/badge/dynamic/json?query=info.requires_python&label=python&url=https%3A%2F%2Fpypi.org%2Fpypi%2Ffasterraster%2Fjson )](https://pypi.python.org/pypi/fasterraster)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)

***fasterraster*** is a fast multi-threaded python library for performing raster operations using [openMP](https://www.openmp.org/) and [numpy](https://numpy.org/) objects complete with simple IO.

 - Github repository: https://github.com/asenogles/fasterraster
 - PyPI: https://pypi.org/project/fasterraster

## Motivation

***fasterraster*** was developed to quickly perform raster operations, enabling self-supervised learning for raster based analyses. ***fasterraster*** provides a cython wrapper for optimized [openMP](https://www.openmp.org/) *c* code. Data objects are handled by [numpy](https://numpy.org/) allowing for straightforward memory management. Currently only computation of visual/morphological features have been implemented however this is open to expansion in the future. All code is still in development and thus it is recommended to test fully before use.

## Installation

***fasterraster*** has currently been tested on Linux and Microsoft windows operating systems. You will need python>=3.6 installed. It is recommended to install ***fasterraster*** within a virtual environment.
### Install using pip

To install ***fasterraster*** from PyPI using pip:

```console
pip install fasterraster
```
### Install from source

To build ***fasterraster*** from source, download this repository and run:
```console
python3 setup.py build_ext --inplace
```
**Note**: You will need to have the required build dependencies installed.

## Example

```python
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

# Time hillshade computation using fasterraster for num-threads
num_threads = [1,2,4,8]
for numt in num_threads:
    time = timeit.timeit(lambda: fr.hillshade_faster_mp(dem.raster, numt), number=NTESTS)
    print(f'hillshade averaged {time/NTESTS:.3f} seconds for {numt} threads')
```
Example output:
```console
python hillshade averaged 2.880 seconds
hillshade averaged 0.081 seconds for 1 threads
hillshade averaged 0.041 seconds for 2 threads
hillshade averaged 0.034 seconds for 4 threads
hillshade averaged 0.024 seconds for 8 threads
```
