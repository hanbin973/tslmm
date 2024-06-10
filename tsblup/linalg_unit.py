import numpy as np
import numba
from numba import prange
from numba import f8, i4
from numba import types, typed
from numba.types import Tuple
from numba.experimental import jitclass

@numba.njit("void(i4[:], i4[:], f8[:])")
def back_solve(Lp, Li, y):
    """
    `L` is lower-triangular Cholesky factor in CSC format: solve `L x = y`.
    `y` is updated in-place.
    """
    x = y
    for j in range(0, x.size):
        for p in range(Lp[j] + 1, Lp[j + 1]):
            x[Li[p]] += x[j]

@numba.njit("void(i4[:], i4[:], f8[:])")
def forward_solve(Lp, Li, y):
    """
    `L` is lower-triangular Cholesky factor in CSC format: solve `L' x = y`.
    `y` is updated in-place.
    """
    x = y 
    for j in range(x.size - 1, -1, -1):
        for p in range(Lp[j] + 1, Lp[j + 1]):
            x[j] += x[Li[p]]

# TODO remove all Lx from the functions and pca class
@numba.njit("void(i4[:], i4[:], f8[:], f8[:,:])")
def back_2dsolve_c(Lp, Li, Lx, Y):
    """
    `L` is lower-triangular Cholesky factor in CSC format: solve `L X = Y`.
    `Y` is updated in-place.
    """
    X = Y
    for j in range(0, X.shape[0]):
        for k in range(X.shape[1]):
            for p in range(Lp[j] + 1, Lp[j + 1]):
                X[Li[p],k] += X[j,k]

@numba.njit("void(i4[:], i4[:], f8[:], f8[:,:])")
def forward_2dsolve_c(Lp, Li, Lx, Y):
    """
    `L` is lower-triangular Cholesky factor in CSC format: solve `L' X = Y`.
    `Y` is updated in-place.
    """
    X = Y
    for j in range(X.shape[0] - 1, -1, -1):
        for k in range(X.shape[1]):
            for p in range(Lp[j] + 1, Lp[j + 1]):
                X[j,k] += X[Li[p],k]

@numba.njit("void(i4[:], i4[:], f8[:], f8[:,:])", parallel=True)
def back_2dsolve_f(Lp, Li, Lx, Y):
    """
    `L` is lower-triangular Cholesky factor in CSC format: solve `L X = Y`.
    `Y` is updated in-place.
    """
    X = Y
    for k in prange(X.shape[1]):
        for j in range(0, X.shape[0]):
            for p in range(Lp[j] + 1, Lp[j + 1]):
                X[Li[p],k] += X[j,k]

@numba.njit("void(i4[:], i4[:], f8[:], f8[:,:])", parallel=True)
def forward_2dsolve_f(Lp, Li, Lx, Y):
    """
    `L` is lower-triangular Cholesky factor in CSC format: solve `L' X = Y`.
    `Y` is updated in-place.
    """
    X = Y
    for k in prange(X.shape[1]):
        for j in range(X.shape[0] - 1, -1, -1):
            for p in range(Lp[j] + 1, Lp[j + 1]):
                X[j,k] += X[Li[p],k]

@numba.njit("Tuple((f8[:,:], f8[:,:]))(f8[:,:])")
def qr(mat):
    return np.linalg.qr(mat)

@numba.njit("f8[::1](i4[::1], i4[::1], i4, f8[::1])")
def csc_v(Ap, Ai, nrow, x):
    """
    `A` is CSC matrix with only ones
    `x` is a vector
    compute y=Ax
    """
    y = np.zeros(nrow)
    for j in range(0, Ap.size - 1):
        jl, ju = Ap[j:j+2]
        for i in Ai[jl:ju]:
            y[i] += x[j]
    return y

@numba.njit("f8[::1](i4[::1], i4[::1], i4, f8[::1])")
def csr_v(Ap, Ai, nrow, x):
    """
    `A` is CSR matrix with only ones
    `x` is a vector
    compute y=Ax
    """
    y = np.zeros(nrow)
    for i in range(0, Ap.size - 1):
        il, iu = Ap[i:i+2]
        for j in Ai[il:iu]:
            y[i] += x[j]
    return y