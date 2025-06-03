from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("color_space_conversion_cython.pyx"),
    include_dirs=[np.get_include()]
) 