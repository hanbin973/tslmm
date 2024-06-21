import numpy as np

import scipy
import scipy.sparse as sparse
from scipy.sparse.linalg import LinearOperator

import numba
from numba import types, typed
from numba.types import f8, i4

from functools import partial

# === matvecs ===
@numba.njit("void(i4[::1], i4[::1], i4, i4, i4[::1], f8[::1,:], f8[:,::1])")
def _sparse_row_dense_col(Ap, Ai, nrow, ncol, rows, X, Y):
    for k in range(X.shape[1]):
        for ii, i in enumerate(rows):
            il, iu = Ap[i:i+2]
            for j in Ai[il:iu]:
                Y[ii,k] += X[j,k] # X가 j 따라 바뀜

def sparse_row_dense_col(Ap, Ai, nrow, ncol, rows, X):
    Y = np.zeros((rows.size, X.shape[1]))
    _sparse_row_dense_col(Ap, Ai, nrow, ncol, rows, X, Y)
    return Y

@numba.njit("void(i4[::1], i4[::1], i4, i4, i4[::1], f8[:,::1], f8[:,::1])")
def _sparse_row_dense_row(Ap, Ai, nrow, ncol, rows, X, Y):
    for ii, i in enumerate(rows):
        il, iu = Ap[i:i+2]
        for j in Ai[il:iu]:
            for k in range(X.shape[1]):
                Y[ii,k] += X[j,k]

def sparse_row_dense_row(Ap, Ai, nrow, ncol, rows, X):
    Y = np.zeros((rows.size, X.shape[1]))
    _sparse_row_dense_row(Ap, Ai, nrow, ncol, rows, X, Y)
    return Y

@numba.njit("void(i4[::1], i4[::1], i4, i4, i4[::1], f8[:,::1], f8[:,::1])")
def _sparse_col_dense_row(Ap, Ai, nrow, ncol, cols, X, Y):
    for jj, j in enumerate(cols):
        jl, ju = Ap[j:j+2]
        for i in Ai[jl:ju]:
            for k in range(X.shape[1]):
                Y[i,k] += X[jj,k]

def sparse_col_dense_row(Ap, Ai, nrow, ncol, cols, X):
    Y = np.zeros((nrow, X.shape[1]))
    _sparse_col_dense_row(Ap, Ai, nrow, ncol, cols, X, Y)
    return Y

@numba.njit("void(i4[::1], i4[::1], i4, i4, i4[::1], f8[::1,:], f8[:,::1])")
def _sparse_col_dense_col(Ap, Ai, nrow, ncol, cols, X, Y):
    for jj, j in enumerate(cols):
        jl, ju = Ap[j:j+2]
        for i in Ai[jl:ju]:
            for k in range(X.shape[1]):
                Y[i,k] += X[jj,k]

def sparse_col_dense_col(Ap, Ai, nrow, ncol, cols, X):
    Y = np.zeros((nrow, X.shape[1]))
    _sparse_col_dense_col(Ap, Ai, nrow, ncol, cols, X, Y)
    return Y

# === triangular solve ===
@numba.njit("void(i4[:], i4[:], f8[:,::1])")
def solve_csr_triu_row(Up, Ui, Y):
    X = Y
    for i in range(X.shape[0]-1, -1, -1):
        il, iu = Up[i:i+2]
        for j in Ui[il+1:iu]:
            for k in range(X.shape[1]):
                X[i,k] += X[j,k]

@numba.njit("void(i4[:], i4[:], f8[:,::1])")
def solve_csc_triu_row(Up, Ui, Y):
    X = Y
    for j in range(X.shape[0]-1, -1, -1):
        jl, ju = Up[j:j+2]
        for i in Ui[jl:ju-1]:
            for k in range(X.shape[1]):
                X[i,k] += X[j,k]
                
@numba.njit("void(i4[:], i4[:], f8[:,::1])")
def solve_csr_tril_row(Lp, Li, Y):
    X = Y
    for i in range(X.shape[0]):
        il, iu = Lp[i:i+2]
        for j in Li[il:iu-1]:
            for k in range(X.shape[1]):
                X[i,k] += X[j,k]

@numba.njit("void(i4[:], i4[:], f8[:,::1])")
def solve_csc_tril_row(Lp, Li, Y):
    X = Y
    for j in range(X.shape[0]):
        jl, ju = Lp[j:j+2]
        for i in Li[jl+1:ju]:
            for k in range(X.shape[1]):
                X[i,k] += X[j,k]

# === basic matrix classes ===
class UnitMatrixCSR():
    """
    CSR matrix. All elements are 1.
    """
    def __init__(self, p, i, shape):
        self.p = p
        self.i = i
        self.shape = shape

    def dot(self, left, id_subset):
        if left.flags.c_contiguous:
            return sparse_row_dense_row(
                self.p,
                self.i,
                self.shape[0],
                self.shape[1],
                id_subset,
                left
            )
        elif left.flags.f_contiguous:
            return sparse_row_dense_col(
                self.p,
                self.i,
                self.shape[0],
                self.shape[1],
                id_subset,
                left
            )
    def tdot(self, right, id_subset):
        if right.flags.c_contiguous:
            return sparse_col_dense_row(
                self.p,
                self.i,
                self.shape[1],
                self.shape[0],
                id_subset,
                right
            )
        elif right.flags.f_contiguous:
            return sparse_col_dense_col(
                self.p,
                self.i,
                self.shape[1],
                self.shape[0],
                id_subset,
                right
            )

class GeomMatrix():
    """
    Upper triangular CSC matrix. Diagonals are all 1. Off-diagonals are all -1.
    `I + A + A^2 + A^3 + ...`
    """
    def __init__(self, p, i, ndim):
        self.p = p
        self.i = i
        self.ndim = ndim
        
    def forward_solve(self, Y):
        """
        T X = Y 
        """
        solve_csc_triu_row(self.p, self.i, Y)

    def back_solve(self, Y):
        """
        T' X = Y
        """
        solve_csr_tril_row(self.p, self.i, Y)

# === design matrix classes ===
def _dot_chr_left(left, design, id_subset, geom, edges_weight_sqrt):
    left *= edges_weight_sqrt[:,None]
    geom.forward_solve(left)
    out = design.dot(left, id_subset)
    #out -= out.sum(axis=0) / out.shape[0] # numba does not support np.mean with axis only for debugging purpose
    return out

def _dot_chr_right(right, design, id_subset, geom, edges_weight_sqrt):
    #right -= right.mean(axis=0)[None,:]
    out = design.tdot(right, id_subset)
    geom.back_solve(out)
    out *= edges_weight_sqrt[:,None]
    return out

class Design:
    def __init__(
        self,
        design_list,
        geom_list,
        edges_area_list,
        n_inds
    ):
        # attributes
        self.design_list = design_list
        self.geom_list = geom_list
        self.edges_area_sqrt_list = [np.sqrt(v) for v in edges_area_list]
        self.n_inds = n_inds
        self.n_chrs = len(edges_area_list)
        self.id_subset = np.arange(n_inds, dtype=np.int32)

        # edge pointer
        self.edge_ptr = np.zeros(self.n_chrs+1, dtype=np.int32)
        self.edge_ptr[1:] = np.cumsum([v.size for v in edges_area_list])

    def dot_left(self, left, id_subset=None):
        if id_subset is None:
            id_subset = self.id_subset
        out = np.zeros((id_subset.size, left.shape[1]))
        for i in range(self.n_chrs):
            il, iu = self.edge_ptr[i:i+2]
            out += _dot_chr_left(
                left[il:iu,:], 
                self.design_list[i],
                id_subset,
                self.geom_list[i],
                self.edges_area_sqrt_list[i]
            )
        return out

    def dot_right(self, right, id_subset=None):
        if id_subset is None:
            id_subset = self.id_subset
        out = np.empty((self.edge_ptr[-1], right.shape[1]))
        for i in range(self.n_chrs):
            il, iu = self.edge_ptr[i:i+2]
            out[il:iu,:] = _dot_chr_right(
                right,
                self.design_list[i],
                id_subset,
                self.geom_list[i],
                self.edges_area_sqrt_list[i]
            )
        return out

    def rsvd(self, n_components=2, n_iter=5, n_oversamples=5, seed=None, id_subset=None):
        np.random.seed(seed)
        random_matrix = np.random.normal(size=(self.edge_ptr[-1], n_components+n_oversamples))
        sample_matrix = self.dot_left(random_matrix, id_subset)
        range_old, _ = np.linalg.qr(sample_matrix)
        for i in range(n_iter):
            range_new, _ = np.linalg.qr(self.dot_right(range_old, id_subset))
            range_old, _ = np.linalg.qr(self.dot_left(range_new, id_subset))
        U, S, V = np.linalg.svd(self.dot_right(range_old, id_subset).T, full_matrices=False)
        return (range_old @ U)[:,:n_components], S[:n_components], V[:n_components,:]    
                
# https://arxiv.org/pdf/2110.02820
# formula 5.3
class RandNystromPreconditioner:
    def __init__(
        self,
        design,
        n_components = 50,
        n_iter = 5,
        seed = None,
        id_subset = None
    ):
        # attributes
        eig_vec, eig_val, _ = design.rsvd(
            n_components=n_components+1,
            n_iter=n_iter,
            seed=seed,
            id_subset=id_subset
        )
        if id_subset.size == n_components:
            self.eig_vec = eig_vec
            self.eig_val = eig_val ** 2
            self.eig_last = 0
        else:
            self.eig_vec = eig_vec[:,:-1]
            self.eig_val = eig_val[:-1] ** 2
            self.eig_last = eig_val[-1] ** 2
    
    # inverse of design
    def dot(self, v, lam):
        v1 = self.eig_vec.T @ v
        v2 = v1 / (lam + self.eig_val)
        out = (lam + self.eig_last) * self.eig_vec @ v2 + v - self.eig_vec @ v1
        out /= lam
        return out


# === CVs ===
def grm_v(v, design, id_subset, geom, edges_weight):
    w = design.tdot(v[:,None], id_subset)
    geom.back_solve(w)
    w *= edges_weight[:,None]
    geom.forward_solve(w)
    return design.dot(w, id_subset).ravel()

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

def covyu_v(v, design, id_subset, geom, edges_weight):
    w = design.tdot(v[:,None], id_subset)
    geom.back_solve(w)
    w *= edges_weight[:,None]
    geom.forward_solve(w)
    return w.ravel()

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

from scipy.sparse.linalg import LinearOperator

def chunk_array(arr, n_chunk):
    chunk_size = int(arr.size / n_chunk)
    return [arr[i:i+chunk_size] for i in range(0, arr.shape[0], chunk_size)]     

class BLUPCrossValidation:
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
        self.n_cv = None

        self.design = Design(
            self.design_list,
            self.geom_list,
            self.edges_area_list,
            self.n_inds
        )

    def set_cv(self, cv=5, n_components=500, n_iter=3):
        # chunk array
        id_inds = np.arange(self.n_inds, dtype=np.int32)
        self.id_chunks = chunk_array(id_inds, cv)

        # construct preconditioner
        self.preconditioners = []
        for id_subset in self.id_chunks:
            self.preconditioners.append(
                RandNystromPreconditioner(
                    self.design,
                    id_subset=np.delete(id_inds, id_subset),
                    n_components=n_components,
                    n_iter=n_iter
                )
            )

        # set cv
        self.n_cv = cv

    def blup_subset(self, id_subset, lam, preconditioner=None):           
        kernel = KernelMatrix(
                self.design_list,
                self.geom_list,
                self.edges_area_list,
                lam,
                id_subset
            )
        kernel_op = LinearOperator(
            (id_subset.size, id_subset.size), 
            matvec=kernel.dot
        )
        covyu = CovOutCumMatrix(
                self.design_list,
                self.geom_list,
                self.edges_area_list,
                id_subset
            )

        if preconditioner is None:
            v1 = sparse.linalg.cg(kernel_op, self.y[id_subset])[0]
            v2 = sparse.linalg.cg(kernel_op, v1)[0]
        else:
            preconditioner_dot = partial(
                preconditioner.dot,
                lam=lam
            )
            preconditioner_op = LinearOperator(
                (id_subset.size, id_subset.size), 
                matvec=preconditioner_dot
            )
            v1 = sparse.linalg.cg(
                kernel_op, 
                self.y[id_subset],
                M=preconditioner_op
            )[0]
            v2 = sparse.linalg.cg(
                kernel_op, 
                v1,
                M=preconditioner_op
            )[0]
        
        return covyu.dot(v1), covyu.dot(v2) 

    def blup(self, lam, id_preconditioner=None):
        return self.compute_base_subset(
            np.arange(self.n_inds, dtype=np.int32), 
            lam,
            preconditioner=None
        )
    
    def cv_error(self, lam):
        assert self.n_cv is not None
        err_nonnormalized, grad = 0, 0
        id_inds = np.arange(self.n_inds, dtype=np.int32)
        for i, id_chunk in enumerate(self.id_chunks):
            id_train, id_test = np.delete(id_inds, id_chunk), id_chunk
            u_blup, gu_blup = self.blup_subset(
                id_train, 
                lam, 
                self.preconditioners[i]
            )            
            y_blup, gy_blup = np.zeros(id_test.size), np.zeros(id_test.size)
            for design in self.design_list:
                y_blup += design.dot(u_blup, id_test).ravel()
                gy_blup += design.dot(gu_blup, id_test).ravel()
            residual = self.y[id_test] - y_blup
            err_nonnormalized += np.sum(residual**2)
            grad += np.dot(residual, gy_blup)
        
        return err_nonnormalized / self.n_inds, 2 * grad / self.n_inds