from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='malvar_he_cutler_cy',
    ext_modules=cythonize("malvar_he_cutler_cy.pyx"),
    include_dirs=[np.get_include()]
) 