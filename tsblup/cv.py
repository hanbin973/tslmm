import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
import numba
from numba import prange
from numba import f8, i4
from numba import types, typed
from numba.types import Tuple
from numba.experimental import jitclass
from .linalg_unit import *

@numba.njit("f8[::1](i4[::1], i4[::1], i4[:], i4, f8[::1])")
def csc_v(Ap, Ai, cols, nrow, x):
    """
    Compute `y=Ax`
    `A` : np.ndarray 
        CSC matrix with only ones
    `x` : np.ndarray 
        vector to multiply
    `cols` :
        columns of `A` to be used
    """
    y = np.zeros(nrow)
    for jj, j in enumerate(cols):
        jl, ju = Ap[j:j+2]
        for i in Ai[jl:ju]:
            y[i] += x[jj]
    return y

@numba.njit("f8[::1](i4[::1], i4[::1], i4[:], i4, f8[::1])")
def csr_v(Ap, Ai, rows, nrow, x):
    """
    `A` is CSR matrix with only ones
    `x` is a vector
    compute y=Ax
    """
    y = np.zeros(rows.size)
    for ii, i in enumerate(rows):
        il, iu = Ap[i:i+2]
        for j in Ai[il:iu]:
            y[ii] += x[j]
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
    
    def dot(self, v, cols):
        assert cols.size == v.size
        return csc_v(self.p, self.i, cols, self.nrow, v)

    def tdot(self, v, rows):
        assert self.nrow == v.size
        return csr_v(self.p, self.i, rows, row.size, v)

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
    
    def dot(self, v, rows):
        assert self.ncol == v.size
        return csr_v(self.p, self.i, rows, rows.size, v)

    def tdot(self, v, cols):
        assert cols.size == v.size
        return csc_v(self.p, self.i, cols, self.ncol, v)

@jitclass(spec_list)
class GeomMatrix():
    """
    Lower-triangular CSC matrix. Diagonals are all 1. Off-diagonals are all -1.
    `I + A + A^2 + A^3 + ...`
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

@numba.njit(f8[::1](f8[::1], UnitMatrixCSR.class_type.instance_type, i4[:], GeomMatrix.class_type.instance_type, f8[::1]))
def grm_v(v, design, id_subset, geom, edges_weight):
    w = design.tdot(v, id_subset)
    geom.back_solve(w)
    w *= edges_weight
    geom.forward_solve(w)
    return design.dot(w, id_subset)

spec_list2 = [
    ('design_list', types.ListType(UnitMatrixCSR.class_type.instance_type)),
    ('geom_list', types.ListType(GeomMatrix.class_type.instance_type)),
    ('edges_weight_list', types.ListType(f8[::1])),
    ('sigma_ep', f8),
    ('id_subset', i4[:])
]

@jitclass(spec_list2)
class KernelMatrix():
    def __init__(
        self,
        design_list,
        geom_list,
        edges_weight_list,
        sigma_ep,
        id_subset
    ):
        assert len(design_list) == len(geom_list)
        assert len(geom_list) == len(edges_weight_list)
        self.design_list = design_list
        self.geom_list = geom_list
        self.edges_weight_list = edges_weight_list
        self.sigma_ep = sigma_ep
        self.id_subset = id_subset

    def dot(self, v):
        w = np.zeros(v.size)
        for i in range(len(self.design_list)):
            w += grm_v(
                v,
                self.design_list[i],
                self.id_subset,
                self.geom_list[i],
                self.edges_weight_list[i]
            )
        return w + self.sigma_ep * v

@numba.njit(f8[::1](f8[::1], UnitMatrixCSR.class_type.instance_type, i4[:], GeomMatrix.class_type.instance_type, f8[::1]))
def covyu_v(v, design, id_subset, geom, edges_weight):
    w = design.tdot(v, id_subset)
    geom.back_solve(w)
    w *= edges_weight
    geom.forward_solve(w)
    return w

@jitclass(spec_list2)
class CovOutCumMatrix():
    def __init__(
        self,
        design_list,
        geom_list,
        edges_weight_list,
        id_subset
    ):
        assert len(design_list) == len(geom_list)
        assert len(geom_list) == len(edges_weight_list)
        self.design_list = design_list
        self.geom_list = geom_list
        self.edges_weight_list = edges_weight_list
        self.id_subset = id_subset

    def dot(self, v):
        w = np.empty(sum([x.size for x in self.edges_weight_list]))
        cnt = 0
        for i in range(len(self.design_list)):
            num_edges = self.edges_weight_list[i].size
            il, iu = cnt, cnt+num_edges
            w[il:iu] = covyu_v(
                v,
                self.design_list[i],
                self.id_subset,
                self.geom_list[i],
                self.edges_weight_list[i]
            )
            cnt += iu
        return w

def chunk_array(arr, n_chunk):
    chunk_size = int(arr.size / n_chunk)
    return [arr[i:i+chunk_size] for i in range(0, arr.shape[0], chunk_size)]     

class BLUP:
    def __init__(
        self,
        design_list,
        geom_list,
        edges_area_list,
        y
    ):
        self.design_list = design_list
        self.geom_list = geom_list
        self.edges_area_list = edges_area_list
        self.y = y

        self.n_inds = y.size
        self.mode = None
        
    def compute_base_subset(self, id_subset, sigma_g, sigma_ep):
        if self.mode is None:
            self.mode = 'base'
        self.edges_weight_list = typed.List([sigma_g * v for v in self.edges_area_list])
            
        kernel = KernelMatrix(
                self.design_list,
                self.geom_list,
                self.edges_weight_list,
                sigma_ep,
                id_subset
            )
        kernel_op = LinearOperator((id_subset.size, id_subset.size), matvec=kernel.dot)
        covyu = CovOutCumMatrix(
                self.design_list,
                self.geom_list,
                self.edges_weight_list,
                id_subset
            )
        return covyu.dot(sparse.linalg.cg(kernel_op, self.y[id_subset])[0])

    def compute_base(self, sigma_g, sigma_ep):
        return self.compute_base_subset(np.arange(self.n_inds, dtype=np.int32), sigma_g, sigma_e)
    
    def cv_error_base(self, sigma_g, sigma_ep, cv=5):
        if self.mode is None:
            self.mode = 'base'
        self.edges_weight_list = typed.List([sigma_g * v for v in self.edges_area_list])
            
        id_inds = np.arange(self.n_inds, dtype=np.int32)
        id_chunks = chunk_array(id_inds, cv)
        err_nonnormalized = 0
        for id_chunk in id_chunks:
            id_train, id_test = np.delete(id_inds, id_chunk), id_chunk
            u_blup = self.compute_base_subset(id_train, sigma_g, sigma_ep)
            y_blup = np.zeros(id_test.size)
            for design in self.design_list:
                y_blup += design.dot(u_blup, id_test)
            err_nonnormalized += np.sum((self.y[id_test] - y_blup)**2)
        return err_nonnormalized / self.n_inds
        
       