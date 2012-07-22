#!/usr/bin/env python

from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy.distutils.misc_util

setup(
        name="astropy-example",
        cmdclass={"build_ext": build_ext},
        ext_modules=[Extension("_cython_demo", ["example.pyx"])],
        include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
    )
