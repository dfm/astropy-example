from __future__ import division
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef extern from "numpy/npy_math.h":
    bint npy_isnan(double x)

cimport cython


@cython.boundscheck(False)  # turn off bounds-checking for entire function
def convolve1d_boundary_wrap(np.ndarray[DTYPE_t, ndim=1] f,
                             np.ndarray[DTYPE_t, ndim=1] g):

    if g.shape[0] % 2 != 1:
        raise ValueError("Convolution kernel must have odd dimensions")

    assert f.dtype == DTYPE and g.dtype == DTYPE

    cdef int nx = f.shape[0]
    cdef int nkx = g.shape[0]
    cdef int wkx = nkx // 2
    cdef np.ndarray[DTYPE_t, ndim=1] fixed = np.empty([nx], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] conv = np.empty([nx], dtype=DTYPE)
    cdef unsigned int i, iii
    cdef int ii

    cdef int iimin, iimax

    cdef DTYPE_t top, bot, ker, val

    # Need a first pass to replace NaN values with value convolved from
    # neighboring values
    for i in range(nx):
        if npy_isnan(f[i]):
            top = 0.
            bot = 0.
            iimin = i - wkx
            iimax = i + wkx + 1
            for ii in range(iimin, iimax):
                iii = ii % nx
                val = f[iii]
                if not npy_isnan(val):
                    ker = g[<unsigned int>(wkx + ii - i)]
                    top += val * ker
                    bot += ker

            if bot != 0.:
                fixed[i] = top / bot
            else:
                fixed[i] = f[i]
        else:
            fixed[i] = f[i]

    # Now run the proper convolution
    for i in range(nx):
        if not npy_isnan(fixed[i]):
            top = 0.
            bot = 0.
            iimin = i - wkx
            iimax = i + wkx + 1
            for ii in range(iimin, iimax):
                iii = ii % nx
                val = fixed[iii]
                ker = g[<unsigned int>(wkx + ii - i)]
                if not npy_isnan(val):
                    top += val * ker
                    bot += ker
            if bot != 0:
                conv[i] = top / bot
            else:
                conv[i] = fixed[i]
        else:
            conv[i] = fixed[i]

    return conv
