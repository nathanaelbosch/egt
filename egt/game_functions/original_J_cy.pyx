"""More efficient version using Cython

Still a WIP!
"""
import cython
import numpy as np
cimport numpy as np
from cython.parallel import prange

from libc.math cimport cos, exp, tanh, fabs, fmax

DTYPE = np.float64
# ctypedef np.float64_t DTYPE_t

from egt.test_functions cimport simple_nonconvex_function_double as f

cdef int alpha = 2


@cython.cdivision(True)
cdef DTYPE_t cython_naive(DTYPE_t x, DTYPE_t u, DTYPE_t x2):
    if x==x2:
        if u==0:
            return 1
        else:
            return 0
    cdef DTYPE_t out
    out = exp(
        -((u - (fmax(0, tanh(3*(f(x) -f(x2)))) * (x2 - x)))**2) /
        (fabs(x-x2) ** alpha))

    return out


"""
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef DTYPE_t [:,:,:] vectorized(DTYPE_t [:] locations, DTYPE_t [:] U):
    cdef int N = locations.shape[0]
    cdef int d = locations.shape[1]
    cdef int i, j, k
    cdef int U_number = U.shape[0]
    #assert d == 1, "Multi dimensional not implemented yet"
    #locations = locations.flatten()
    #U = U.flatten()

    cdef DTYPE_t [:,:,:] out = np.empty((N, N, U.shape[0]))
    #cdef DTYPE_t out[N][N][U_number]

    for k in prange(U_number, nogil=True):
        for i in range(N):
            for j in range(N):
                out[i, j, k] = cython_naive(
                    locations[i], U[k], locations[j])
    return out
"""