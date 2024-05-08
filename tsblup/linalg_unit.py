import numpy as np
import numba
from numba import prange
from numba.types import Tuple

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
