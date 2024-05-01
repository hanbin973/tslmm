import numba
import numpy as np
import scipy.sparse as sparse
import scipy.io as io
import time

# --- lib

@numba.njit("void(i4[:], i4[:], f8[:], f8[:])")
def back_solve(Lp, Li, Lx, y):
    """
    `L` is lower-triangular Cholesky factor in CSC format: solve `L x = y`.
    `y` is updated in-place.
    """
    x = y
    for j in range(0, x.size):
        x[j] /= Lx[Lp[j]]
        for p in range(Lp[j] + 1, Lp[j + 1]):
            x[Li[p]] -= Lx[p] * x[j]

@numba.njit("void(i4[:], i4[:], f8[:], f8[:])")
def forward_solve(Lp, Li, Lx, y):
    """
    `L` is lower-triangular Cholesky factor in CSC format: solve `L' x = y`.
    `y` is updated in-place.
    """
    x = y
    for j in range(x.size - 1, -1, -1):
        for p in range(Lp[j] + 1, Lp[j + 1]):
            x[j] -= Lx[p] * x[Li[p]]
        x[j] /= Lx[Lp[j]]

@numba.njit("void(i4, i4, i4[:], i4[:], f8[:], i4[:], i4[:], f8[:,:], f8[:])")
def _factorize_worker(block_start, block_size, Lp, Li, Lx, Zp, Zi, P, D):
    """
    `L` is n-by-n sparse lower-triangular Cholesky factor in CSC format.
    `Z` is n-by-m sparse design matrix in CSC format.

    Diagonalize a block of `Z (LL')^{-1} Z'`.

    Fills P and D so that, `P[block_start:block_start+block_size]` and
    `D[block_start:block_start+block_size]` are the eigenvectors and
    eigenvalues.
    """
    n, s = block_start, block_size
    x = np.zeros(Lp.size - 1)
    output = np.empty((s, s))
    ii = 0
    for i in range(n, n + s):
        il, iu = Zp[i:i+2]
        x[Zi[il:iu]] = 1.0
        back_solve(Lp, Li, Lx, x)
        forward_solve(Lp, Li, Lx, x)
        jj = 0
        for j in range(n, n + s):
            jl, ju = Zp[j:j+2]
            output[ii, jj] = np.sum(x[Zi[jl:ju]])
            jj += 1
        x[:] = 0.0
        ii += 1
    D[n:n+s], P[n:n+s,:s] = np.linalg.eig(output)

@numba.njit("Tuple((f8[:, :], f8[:]))(i4, i4[:], i4[:], f8[:], i4[:], i4[:])")
def _factorize_parallel(blocks, Lp, Li, Lx, Zp, Zi):
    """
    `L` is n-by-n sparse lower-triangular Cholesky factor in CSC format.
    `Z` is n-by-m sparse design matrix in CSC format.

    Factorize random effects contribution to covariance matrix across blocks.

    The eigenvectors are stored in a tensor left-indexed by block. As the size
    of each block may vary slightly, this tensor is padded with np.nan.

    The eigenvalues are stored in a matrix left-indexed by block, that is padded
    with np.nan.
    """
    obs_dim = Zp.size - 1
    block_coord = np.linspace(0, obs_dim, blocks + 1).astype(np.int32)
    block_size = np.diff(block_coord)
    assert np.all(block_size > 1), "Use fewer blocks"
    output_dim = np.max(block_size)
    eigvec = np.zeros((obs_dim, output_dim))
    eigval = np.zeros(obs_dim)
    for i in range(blocks): # TODO: parallelize with prange
        n, s = block_coord[i], block_size[i]
        _factorize_worker(n, s, Lp, Li, Lx, Zp, Zi, eigvec, eigval)
    return eigvec, eigval

# --- implm

Tt = io.mmread("T.mtx").T.tocsc() #lower tri
Zt = io.mmread("Z.mtx").T.tocsc()

sigma = 0.25
tau = 0.6
y = np.ones(Zt.shape[1])

# a single block (all samples)
st = time.time()
P, D = _factorize_parallel(1, Tt.indptr, Tt.indices, Tt.data, Zt.indptr, Zt.indices)
en = time.time()
print(f"Timing {en - st}")

inv = P @ np.diag(1.0 / (D * tau + sigma)) @ P.T # precision matrix (not needed)
resid = (P.T @ y) / np.sqrt(D * tau + sigma) # quadratic in loglik
logdet = np.sum(np.log(D * tau + sigma)) # logdet in loglik

# check
S = Zt.T @ sparse.linalg.spsolve(Tt @ Tt.T, np.asarray(Zt.todense()))
inv_check = np.linalg.inv(S * tau + sigma * np.eye(S.shape[0]))
np.testing.assert_allclose(inv, inv_check)

resid_check = np.linalg.cholesky(inv_check).T @ y
np.testing.assert_allclose(np.sum(resid ** 2), np.sum(resid_check ** 2))

_, logdet_check = np.linalg.slogdet(inv_check)
np.testing.assert_allclose(logdet, -logdet_check)

