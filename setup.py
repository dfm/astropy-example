#!/usr/bin/env python

from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy.distutils.misc_util


c_ext = [Extension("_cython", ["_cython.pyx"]),
         Extension("_convolution", ["_convolution.c"])]


setup(
        name="astropy-example",
        cmdclass={"build_ext": build_ext},
        ext_modules=c_ext,
        include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
    )
