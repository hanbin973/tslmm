import numpy as np
import numba

from .linalg_unit import *


## TODO jitclass
"""
class RowEdgeDesign:
    def __init__(self, Z, T, w):
        """
        Z is the design matrix in CSR
        T is a CSR upper triangular matrix
        w is a vector of edge weights
        """

        self.Zp = Z.indptr
        self.Zi = Z.indices
        self.Zx = Z.data

        self.Tp = T.indptr
        self.Ti = T.indices
        self.Tx = T.data

        self.w = w
        
        self.num_rows = Z.shape[0]
        self.num_edges = Z.shape[1]
"""

@numba.njit("f8(i4, i4, i4[:], i4[:], i4[:], i4[:], f8[:], f8[:])")   
def _trait2_worker(block_start, block_size, Lp, Li, Zp, Zi, w, y):
    """
    `L` is n-by-n sparse lower triangular Cholesky factor in CSC format
    `Z` is n-by-m sparse design matrix  in CSR format

    Compute y' * GRM * y
    """
    
    n, s = block_start, block_size
    w_sqrt = np.sqrt(w) 

    x = csc_v(Zp[n:n+s+1], Zi, Lp.size - 1, y[n:n+s])
    back_solve(Lp, Li, x)
    x *= w_sqrt
    
    return np.sum(x ** 2)

@numba.njit("f8(i4, i4, i4[:], i4[:], i4[:], i4[:], f8[:])")   
def _trace_worker(block_start, block_size, Lp, Li, Zp, Zi, w):
    """
    `L` is n-by-n sparse lower triangular Cholesky factor in CSC format
    `Z` is n-by-m sparse design matrix  in CSR format

    Compute the trace of GRM
    """
    n, s = block_start, block_size
    w_sqrt = np.sqrt(w)
    x = np.zeros(Lp.size - 1)
    
    out = 0 
    for i in range(n, n + s):
        il, iu = Zp[i:i+2]
        x[Zi[il:iu]] = 1.0
        back_solve(Lp, Li, x)
        x *= w_sqrt
        out += np.sum(x ** 2)
        x[:] = 0
    
    return out

@numba.njit("f8(i4, i4, i4[:], i4[:], i4[:], i4[:], f8[:])")   
def _trace2_worker(block_start, block_size, Lp, Li, Zp, Zi, w):
    """
    `L` is n-by-n sparse lower triangular Cholesky factor in CSC format
    `Z` is n-by-m sparse design matrix  in CSR format

    Compute the trace of GRM^2
    """
    n, s = block_start, block_size
    x = np.zeros(Lp.size - 1)
    
    out = 0 
    for i in range(n, n + s):
        il, iu = Zp[i:i+2]
        x[Zi[il:iu]] = 1.0
        back_solve(Lp, Li, x)
        x *= w
        forward_solve(Lp, Li, x)
        for j in range(n, n + s):
            jl, ju = Zp[j:j+2]
            out += np.sum(x[Zi[jl:ju]]) ** 2
        x[:] = 0
    
    return out

        

