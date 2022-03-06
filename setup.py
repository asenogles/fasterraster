import os
import sys
import numpy

from setuptools import setup, find_packages, Extension
from Cython.Build import build_ext

if sys.platform == 'linux':
    print('compiling for linux using gcc')
    compile_args = ['-fopenmp', '-std=c17', '-Wformat-truncation=0']
    link_args=['-fopenmp']
elif sys.platform == 'win32':
    print('compiling for windows')
    compile_args = ['-openmp']
    link_args = []
else:
    print(f'Do not recognise system platform: {sys.platform}')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

extensions = [
    Extension(
        name='fasterraster._ext',
        sources=[
            'fasterraster/_ext.pyx', 
            'fasterraster/bil/bil.c',
            'fasterraster/flo/flo.c',
            'fasterraster/npy/npy.c',
            'fasterraster/operations/operations.c'
            ],
        include_dirs=[numpy.get_include()],
        extra_compile_args = compile_args,
        extra_link_args = link_args,
        )
]


# Make sure everything is compiled with pyton 3
for e in extensions:
    e.cython_directives = {'language_level': '3'}

# load readme
with open('README.md', mode = 'rb') as readme_file:
    readme = readme_file.read().decode('utf-8')

setup(
    name = 'fasterraster',
    version='0.0.2',
    description='Fast multi-threaded raster operations with simple IO',
    long_description=readme,
    long_description_content_type='text/markdown',
    url='https://github.com/asenogles/fasterraster',
    author='asenogles',
    license='BSD',
    packages=find_packages(),
    install_requires=[
        "cython>=0.29.21",
        "numpy>=1.19.0",
        "scipy>=1.7.0",
        "tifffile"
        ],
    python_requires='>=3.6',
    cmdclass = {'build_ext': build_ext},
    ext_modules = extensions,
)