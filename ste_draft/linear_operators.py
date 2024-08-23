"""
The idea:

    - Ignoring fixed effects for now, the model is `y = Zu + e` for `u ~ N(0, tQ)` and  `e ~ N(0, sI)`
    - Where `Q = (LL')^{-1}` with very sparse lower-triangular `L`; `Z` is also very sparse
    - Thus `y ~ N(0, S)` with `S = tZQZ' + sI`, for variance components `s, t`
    - Can evaluate matvec products `Sx` very quickly via forward/backward substitution
    - We're assuming that 0 << rank(S) << rank(Q)

The gradient of the deviance wrt `s, t` is:

    - `dloss/ds = tr(S^{-1}) - y' S^{-1} S^{-1} y`
    - `dloss/dt = tr(S^{-1} Q) - y' S^{-1} Q S^{-1} y`

We can get a stochastic estimate for the gradient by:

    - Can estimate `tr(S^{-1})` via Monte Carlo, by `mean(x' S^{-1} x)` for random isotropic `x`
    - Can solve `S^{-1} x = b` by preconditioned CG (using a random low-rank preconditioner)
    - The preconditioner can in theory be used to reduce MC variance of trace, via a control variate
    - The preconditioner can be reused for different `s, t` and is embarrassingly parallel to construct
    - Thus it's worthwhile to keep the rank of the preconditioner as large as possible

Then use stochastic gradient descent to optimize `log(s), log(t)`:

    - End up with samples of the gradient around the mode
    - To estimate the mode, fit a quadratic using these samples
    - That is, fit a linear regression to the (noisy) gradient samples; the
      mode is (approx) where the fitted line crosses 0

Riffing on: https://proceedings.mlr.press/v162/wenger22a/wenger22a.pdf
"""

import tskit
import numba
import numpy as np
import ray
import scipy.sparse as sparse
import scipy
import sys
import os

tsblup_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(tsblup_path)
import tsblup.operations as operations
import tsblup.matrices as matrices


# --- linear operators --- #

_i1r = numba.types.Array(numba.types.int32, 1, 'C', readonly=True)
_f1r = numba.types.Array(numba.types.float64, 1, 'C', readonly=True)
_f2r = numba.types.Array(numba.types.float64, 2, 'C', readonly=True)
_f2w = numba.types.Array(numba.types.float64, 2, 'C', readonly=False)


class TraitCovariance:
    """
    Covariance contribution from tree sequence: given `y = Z u + e` where `u ~
    N(0, tau L^{-1} L^{-T})` and `e ~ N(0, sigma I)`, then `y ~ N(0, tau Z
    L^{-1} L^{-T} Z^T + sigma I)`
    """

    @staticmethod
    @numba.njit(_f2w(_i1r, _i1r, _f1r, _f2r), parallel=True)
    def backward_solve(Lp, Li, Lx, y):
        """
        `L` is lower-triangular Cholesky factor in CSC format: solve `L x = y`.
        `y` is updated in-place.
        """
        r, c = y.shape
        x = y.copy()
        for i in numba.prange(c):
            for j in range(0, r):
                x[j, i] /= Lx[Lp[j]]
                for p in range(Lp[j] + 1, Lp[j + 1]):
                    x[Li[p], i] -= Lx[p] * x[j, i]
        #for j in range(0, r):  # why is this so much slower?
        #   x[j] /= Lx[Lp[j]]
        #   for p in range(Lp[j] + 1, Lp[j + 1]):
        #       x[Li[p]] -= Lx[p] * x[j]
        return x

    @staticmethod
    @numba.njit(_f2w(_i1r, _i1r, _f1r, _f2r), parallel=True)
    def forward_solve(Lp, Li, Lx, y):
        """
        `L` is lower-triangular Cholesky factor in CSC format: solve `L' x = y`.
        `y` is updated in-place.
        """
        r, c = y.shape
        x = y.copy()
        for i in numba.prange(c):
            for j in range(r - 1, -1, -1):
                for p in range(Lp[j] + 1, Lp[j + 1]):
                    x[j, i] -= Lx[p] * x[Li[p], i]
                x[j, i] /= Lx[Lp[j]]
        #for j in range(r - 1, -1, -1):  # why is this so much slower?
        #   for p in range(Lp[j] + 1, Lp[j + 1]):
        #       x[j] -= Lx[p] * x[Li[p]]
        #   x[j] /= Lx[Lp[j]]
        return x

    def __init__(self, tree_sequence, mutation_rate=1.0):
        ts = operations.split_upwards(tree_sequence)
        mutational_target = mutation_rate * (ts.edges_right - ts.edges_left) * \
            (ts.nodes_time[ts.edges_parent] - ts.nodes_time[ts.edges_child])
        self.dim = ts.num_individuals
        self.kernel_dim = ts.num_edges
        self.Z = matrices.edge_individual_matrix(ts).T
        self.L = scipy.sparse.identity(ts.num_edges) - matrices.edge_adjacency(ts).T
        self.L = self.L.T @ scipy.sparse.diags_array(1 / np.sqrt(mutational_target))
        self.L.sort_indices()

    def __call__(self, sigma, tau, y):
        """
        Matrix-vector product
        """
        # TODO can we do this without forming num_edges x y.shape[1] vectors -- Peter thinks yes
        assert self.L.has_sorted_indices
        is_vector = y.ndim == 1
        if is_vector: y = y.reshape(-1, 1)
        Zy = self.Z.T @ y
        Zy = self.backward_solve(self.L.indptr, self.L.indices, self.L.data, Zy)
        Zy = self.forward_solve(self.L.indptr, self.L.indices, self.L.data, Zy)
        Zy = self.Z @ Zy * tau + y * sigma
        if is_vector: Zy = Zy.squeeze()
        return Zy


    def solve(self, sigma, tau, b, preconditioner=None, maxitt=None, atol=0., rtol=1e-5):
        """
        Matrix-vector product with inverse covariance matrix.

        Re-implementing CG here to parallelise with multiple rhs, adapted from
        https://github.com/scipy/scipy/blob/v1.14.1/scipy/sparse/linalg/_isolve/iterative.py#L283-L388
        """
        is_vector = b.ndim == 1
        assert b.shape[0] == self.dim
        assert atol >= 0 and rtol >= 0
        if maxitt is None: maxitt = self.dim
        if not b.any(): return b, 0, True  # b = 0
        if is_vector: b = b.reshape(-1, 1)
        atol = max(atol, rtol * np.linalg.norm(b))
        M = preconditioner
        x = np.zeros_like(b) if M is None else M(b)
        r = b - self(sigma, tau, x)
        for itt in range(maxitt):
            if np.linalg.norm(r) < atol:  # frobenius norm
                break
            z = r if M is None else M(r)
            rho = np.sum(r * z, axis=0)
            if itt == 0:
                p = z.copy()
            else:
                beta = rho / last_rho
                p *= beta[np.newaxis]
                p += z
            q = self(sigma, tau, p)
            alpha = rho / np.sum(p * q, axis=0)
            x += alpha[np.newaxis] * p
            r -= alpha[np.newaxis] * q
            last_rho = rho
        success = itt + 1 < maxitt
        if is_vector: x = x.squeeze()
        return x, itt, success

    def as_matrix(self, sigma, tau):
        """
        For testing only
        """
        assert self.L.has_sorted_indices
        Zd = np.asarray(self.Z.todense())
        Sigma = Zd.T
        Sigma = self.backward_solve(self.L.indptr, self.L.indices, self.L.data, Sigma)
        Sigma = self.forward_solve(self.L.indptr, self.L.indices, self.L.data, Sigma)
        Sigma = tau * Zd @ Sigma + sigma * np.eye(self.dim)
        return Sigma

    def simulate(self, sigma, tau, rng):
        """
        Returns edge effects, genetics values, observed values
        """
        u = rng.normal(size=(self.kernel_dim, 1))
        r = self.forward_solve(self.L.indptr, self.L.indices, self.L.data, u).flatten() * np.sqrt(tau)
        x = self.Z @ r
        e = rng.normal(size=self.dim) * np.sqrt(sigma)
        return r, x, x + e


class NystromPreconditioner:
    """
    Randomized Nystrom approximation to the inverse square root of the
    covariance matrix defined by `TraitCovariance`
    """

    @staticmethod
    def _rand_eigh(covariance, rank, samples, seed):
        """
        Algorithm 16 in https://arxiv.org/pdf/2002.01387
        """
        assert samples >= rank > 0
        rng = np.random.default_rng(seed)
        test_vectors = rng.normal(size=(covariance.dim, samples))
        test_vectors = np.linalg.qr(test_vectors).Q  # orthonormalise
        proj_vectors = covariance(0, 1, test_vectors)
        shift = np.sqrt(covariance.dim) * np.spacing(np.linalg.norm(proj_vectors))
        proj_vectors += shift * test_vectors
        chol_factor = np.linalg.cholesky(test_vectors.T @ proj_vectors)
        coef_matrix = np.linalg.solve(chol_factor, proj_vectors.T)
        U, D, _ = np.linalg.svd(coef_matrix.T, full_matrices=False)
        D = np.maximum(D ** 2 - shift, 0)
        return D[:rank], U[:, :rank]

    def __init__(self, covariance, rank, samples=None, seed=None):
        samples = rank if samples is None else max(rank, samples)
        self.dim = covariance.dim
        self.D, self.U = self._rand_eigh(covariance, rank, samples, seed)

    def __call__(self, sigma, tau, y):
        """
        Inverse-vector product: section 17.2 in https://arxiv.org/pdf/2002.01387
        """
        is_vector = y.ndim == 1
        if is_vector: y = y.reshape(-1, 1)
        S = (self.D * tau + sigma) / (np.min(self.D) * tau + sigma)
        My = (self.U.T @ y) * (1 - 1 / S)[:, np.newaxis]
        My = y - self.U @ My
        if is_vector: My = My.squeeze()
        return My

    def as_matrix(self, sigma, tau):
        """
        For testing only
        """
        S = (self.D * tau + sigma) / (np.min(self.D) * tau + sigma)
        return np.eye(self.dim) - self.U @ np.diag(1 - S) @ self.U.T

    def logdet(self, sigma, tau):
        S = (self.D * tau + sigma) / (np.min(self.D) * tau + sigma)
        return np.sum(np.log(S))

    def grad_sigma(self, sigma, tau, y=None):
        """
        return trace(solve(preconditioner, deriv(preconditioner, sigma)))
        or if vectors `y` are passed:
        return preconditioner @ deriv(preconditioner, sigma) @ y
        """
        spectrum = self.D * tau + sigma
        norm = np.min(spectrum)
        spectrum /= norm
        grad_sigma = (1 - spectrum) / spectrum / norm
        if y is None:
            return np.sum(grad_sigma)
        assert y.ndim == 2
        control = (self.U.T @ y) * grad_sigma[:, np.newaxis]
        return self.U @ control

    def grad_tau(self, sigma, tau, y=None):
        """
        return trace(solve(preconditioner, deriv(preconditioner, tau)))
        or if vectors `y` are passed:
        return preconditioner @ deriv(preconditioner, tau) @ y
        """
        spectrum = self.D * tau + sigma
        norm = np.min(spectrum)
        spectrum /= norm
        grad_tau = (self.D - np.min(self.D) * spectrum) / spectrum / norm
        if y is None:
            return np.sum(grad_tau)
        assert y.ndim == 2
        control = (self.U.T @ y) * grad_tau[:, np.newaxis]
        return self.U @ control
