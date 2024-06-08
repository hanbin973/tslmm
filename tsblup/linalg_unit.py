import numpy as np
import numba
from numba import prange
from numba.types import Tuple

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

# === Experimental for CV-VC ===
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

spec_list = [
    ('p', i4[::1]),
    ('i', i4[::1]), 
    ('nrow', i4), 
    ('ncol', i4),
    ('ndim', i4)
]

@jitclass(spec_list)
class UnitMatrixCSC():
    """
    CSC matrix. All elements are 1.
    """
    def __init__(self, p, i, nrow, ncol):
        self.p = p
        self.i = i
        self.nrow = nrow
        self.ncol = ncol
    
    def dot(self, v):
        assert self.ncol == v.size
        return csc_v(self.p, self.i, self.nrow, v)

    def tdot(self, v):
        assert self.nrow == v.size
        return csr_v(self.p, self.i, self.ncol, v)

@jitclass(spec_list)
class UnitMatrixCSR():
    """
    CSR matrix. All elements are 1.
    """
    def __init__(self, p, i, nrow, ncol):
        self.p = p
        self.i = i
        self.nrow = nrow
        self.ncol = ncol
    
    def dot(self, v):
        assert self.ncol == v.size
        return csr_v(self.p, self.i, self.nrow, v)

    def tdot(self, v):
        assert self.nrow == v.size
        return csc_v(self.p, self.i, self.ncol, v)

@jitclass(spec_list)
class GeomMatrix():
    """
    Lower-triangular CSC matrix. Diagonals are all 1. Off-diagonals are all -1.
    """
    def __init__(self, p, i, ndim):
        self.p = p
        self.i = i
        self.ndim = ndim
        
    def back_solve(self, y):
        """
        L x =y 
        """
        back_solve(self.p, self.i, y)

    def forward_solve(self, y):
        """
        L' x = y
        """
        forward_solve(self.p, self.i, y)

@numba.njit(f8[::1](f8[::1], UnitMatrixCSR.class_type.instance_type, GeomMatrix.class_type.instance_type, f8[::1]), fastmath=True)
def grm_v(v, design, geom, edges_weight):
    w = design.tdot(v)
    geom.back_solve(w)
    w *= edges_weight
    geom.forward_solve(w)
    return design.dot(w)

spec_list2 = [
    ('design_list', types.ListType(UnitMatrixCSR.class_type.instance_type)),
    ('geom_list', types.ListType(GeomMatrix.class_type.instance_type)),
    ('edges_weight_list', types.ListType(f8[::1])),
    ('sigma_ep', f8)
]

@jitclass(spec_list2)
class KernelMatrix():
    def __init__(
        self,
        design_list,
        geom_list,
        edges_weight_list,
        sigma_ep
    ):
        assert len(design_list) == len(geom_list)
        assert len(geom_list) == len(edges_weight_list)
        self.design_list = design_list
        self.geom_list = geom_list
        self.edges_weight_list = edges_weight_list
        self.sigma_ep = sigma_ep

    def dot(self, v):
        w = np.zeros(v.size)
        for i in range(len(self.design_list)):
            w += grm_v(
                v,
                self.design_list[i],
                self.geom_list[i],
                self.edges_weight_list[i]
            )
            
        return w + self.sigma_ep * v
