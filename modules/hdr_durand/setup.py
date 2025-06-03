from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="hdr_durand_cy",
    ext_modules=cythonize("hdr_durand_cy.pyx"),
    include_dirs=[np.get_include()],
    zip_safe=False,
) 