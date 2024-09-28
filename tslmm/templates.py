import numpy as np

import scipy
import scipy.sparse as sparse

import numba

# === in-place linalg of unit matrices ===
@numba.njit("void(i4[::1], i4[::1], i4, i4, i4[::1], f8[::1,:], f8[:,::1])")
def _sparse_row_dense_col(Ap, Ai, nrow, ncol, rows, X, Y):
    for k in range(X.shape[1]):
        for ii, i in enumerate(rows):
            il, iu = Ap[i:i+2]
            for j in Ai[il:iu]:
                Y[ii,k] += X[j,k] # X가 j 따라 바뀜

@numba.njit("void(i4[::1], i4[::1], i4, i4, i4[::1], f8[:,::1], f8[:,::1])")
def _sparse_row_dense_row(Ap, Ai, nrow, ncol, rows, X, Y):
    for ii, i in enumerate(rows):
        il, iu = Ap[i:i+2]
        for j in Ai[il:iu]:
            for k in range(X.shape[1]):
                Y[ii,k] += X[j,k]


@numba.njit("void(i4[::1], i4[::1], i4, i4, i4[::1], f8[:,::1], f8[:,::1])")
def _sparse_col_dense_row(Ap, Ai, nrow, ncol, cols, X, Y):
    for jj, j in enumerate(cols):
        jl, ju = Ap[j:j+2]
        for i in Ai[jl:ju]:
            for k in range(X.shape[1]):
                Y[i,k] += X[jj,k]


@numba.njit("void(i4[::1], i4[::1], i4, i4, i4[::1], f8[::1,:], f8[:,::1])")
def _sparse_col_dense_col(Ap, Ai, nrow, ncol, cols, X, Y):
    for jj, j in enumerate(cols):
        jl, ju = Ap[j:j+2]
        for i in Ai[jl:ju]:
            for k in range(X.shape[1]):
                Y[i,k] += X[jj,k]

@numba.njit("void(i4[:], i4[:], f8[:,::1])")
def _solve_csr_triu_row(Up, Ui, Y):
    X = Y
    for i in range(X.shape[0]-1, -1, -1):
        il, iu = Up[i:i+2]
        for j in Ui[il+1:iu]:
            for k in range(X.shape[1]):
                X[i,k] += X[j,k]

@numba.njit("void(i4[:], i4[:], f8[:,::1])")
def _solve_csc_triu_row(Up, Ui, Y):
    X = Y
    for j in range(X.shape[0]-1, -1, -1):
        jl, ju = Up[j:j+2]
        for i in Ui[jl:ju-1]:
            for k in range(X.shape[1]):
                X[i,k] += X[j,k]
                
@numba.njit("void(i4[:], i4[:], f8[:,::1])")
def _solve_csr_tril_row(Lp, Li, Y):
    X = Y
    for i in range(X.shape[0]):
        il, iu = Lp[i:i+2]
        for j in Li[il:iu-1]:
            for k in range(X.shape[1]):
                X[i,k] += X[j,k]

@numba.njit("void(i4[:], i4[:], f8[:,::1])")
def _solve_csc_tril_row(Lp, Li, Y):
    X = Y
    for j in range(X.shape[0]):
        jl, ju = Lp[j:j+2]
        for i in Li[jl+1:ju]:
            for k in range(X.shape[1]):
                X[i,k] += X[j,k]

# === linalg of unit matrices ===
def sparse_row_dense_col(Ap, Ai, nrow, ncol, rows, X):
    Y = np.zeros((rows.size, X.shape[1]))
    _sparse_row_dense_col(Ap, Ai, nrow, ncol, rows, X, Y)
    return Y

def sparse_row_dense_row(Ap, Ai, nrow, ncol, rows, X):
    Y = np.zeros((rows.size, X.shape[1]))
    _sparse_row_dense_row(Ap, Ai, nrow, ncol, rows, X, Y)
    return Y

def sparse_col_dense_row(Ap, Ai, nrow, ncol, cols, X):
    Y = np.zeros((nrow, X.shape[1]))
    _sparse_col_dense_row(Ap, Ai, nrow, ncol, cols, X, Y)
    return Y

def sparse_col_dense_col(Ap, Ai, nrow, ncol, cols, X):
    Y = np.zeros((nrow, X.shape[1]))
    _sparse_col_dense_col(Ap, Ai, nrow, ncol, cols, X, Y)
    return Y

# === unit matrix templates ===
class UnitMatrixCSR():
    """
    CSR matrix M. All elements are 1.
    Supports M @ v (dot) and M' @ v (tdot).
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
        _solve_csc_triu_row(self.p, self.i, Y)

    def back_solve(self, Y):
        """
        T' X = Y
        """
        _solve_csr_tril_row(self.p, self.i, Y)

# === matrics in y = Ar + e ===
class DesignNormed:
    """
    Zc * Uc * Dc^{1/2} where D is the diagonal matrix of edge variances.
    The matrix is normalized to have row-mean zero.
    """
    def __init__(
        self,
        design_list,
        geom_list,
        edges_area_list,
        num_individuals
        ):

        # attributes
        self.design_list = design_list
        self.geom_list = geom_list
        self.edges_area_sqrt_list = [np.sqrt(v) for v in edges_area_list]
        self.nun_individuals = num_individuals
        self.num_chromosomes = len(edges_area_list)
        self.id_subset = np.arange(num_individuals, dtype=np.int32)

        # edge pointer
        self.edge_ptr = np.zeros(self.num_chromosomes+1, dtype=np.int32)
        self.edge_ptr[1:] = np.cumsum([v.size for v in edges_area_list])
    
    # numba does not support np.mean with axis
    @staticmethod
    def _dot_left_chromosome(
            left, design, id_subset,
            geom, edges_weight_sqrt
            ):
        left *= edges_weight_sqrt[:,None]
        geom.forward_solve(left)
        out = design.dot(left, id_subset)
        out -= out.sum(axis=0) / out.shape[0]     
        return out

    @staticmethod
    def _dot_right_chromosome(
            right, design, id_subset, 
            geom, edges_weight_sqrt
            ):
        right -= right.sum(axis=0)[None,:] / right.shape[0]
        out = design.tdot(right, id_subset)
        geom.back_solve(out)
        out *= edges_weight_sqrt[:,None]
        return out

    def dot_left(self, left, id_subset=None):
        if id_subset is None:
            id_subset = self.id_subset
        out = np.zeros((id_subset.size, left.shape[1]))
        for i in range(self.num_chromosomes):
            il, iu = self.edge_ptr[i:i+2]
            out += self._dot_left_chromosome(
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
        for i in range(self.num_chromosomes):
            il, iu = self.edge_ptr[i:i+2]
            out[il:iu,:] = self._dot_right_chromosome(
                    right,
                    self.design_list[i],
                    id_subset,
                    self.geom_list[i],
                    self.edges_area_sqrt_list[i]
                    )
        return out

    def rsvd(self, n_components=2, n_iter=3, n_oversamples=5, seed=None, id_subset=None):
        np.random.seed(seed)
        random_matrix = np.random.normal(
                size=(self.edge_ptr[-1], n_components+n_oversamples)
                )
        sample_matrix = self.dot_left(random_matrix, id_subset)
        range_old, _ = np.linalg.qr(sample_matrix)
        for i in range(n_iter-1):
            range_new, _ = np.linalg.qr(self.dot_right(range_old, id_subset))
            range_old, _ = np.linalg.qr(self.dot_left(range_new, id_subset))
        U, S, V = np.linalg.svd(
                self.dot_right(range_old, id_subset).T,
                full_matrices=False
                )
        return (range_old @ U)[:,:n_components], S[:n_components], V[:n_components,:]
