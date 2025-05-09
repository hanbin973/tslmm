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

from __future__ import annotations

import tskit
import numba
import numpy as np
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
    Covariance between observations given `y = Z u + e` where `u ~
    N(0, tau L^{-1} L^{-T})` and `e ~ N(0, sigma I)`, then `y ~ N(0, tau Z
    L^{-1} L^{-T} Z^T + sigma I)`
    """

    @staticmethod
    @numba.njit(_f2w(_i1r, _i1r, _f1r, _f2r), parallel=True)
    def backward_solve(Lp, Li, Lx, y):
        """
        `L` is lower-triangular Cholesky factor in CSC format: solve `L x = y`.
        """
        r, c = y.shape
        x = y.copy()
        for i in numba.prange(c):
            for j in range(0, r):
                x[j, i] /= Lx[Lp[j]]
                for p in range(Lp[j] + 1, Lp[j + 1]):
                    x[Li[p], i] -= Lx[p] * x[j, i]
        return x

    @staticmethod
    @numba.njit(_f2w(_i1r, _i1r, _f1r, _f2r), parallel=True)
    def forward_solve(Lp, Li, Lx, y):
        """
        `L` is lower-triangular Cholesky factor in CSC format: solve `L' x = y`.
        """
        r, c = y.shape
        x = y.copy()
        for i in numba.prange(c):
            for j in range(r - 1, -1, -1):
                for p in range(Lp[j] + 1, Lp[j + 1]):
                    x[j, i] -= Lx[p] * x[Li[p], i]
                x[j, i] /= Lx[Lp[j]]
        return x

    def __init__(self, tree_sequence: tskit.TreeSequence, mutation_rate: float = 1.0):
        ts = operations.split_upwards(tree_sequence)
        self.dim = ts.num_individuals
        self.factor_dim = ts.num_edges
        self.D = mutation_rate * (ts.edges_right - ts.edges_left) * \
            (ts.nodes_time[ts.edges_parent] - ts.nodes_time[ts.edges_child])
        self.Z = matrices.edge_individual_matrix(ts).T
        self.L = scipy.sparse.identity(ts.num_edges) - matrices.edge_adjacency(ts).T
        self.L = self.L.T @ scipy.sparse.diags_array(1 / np.sqrt(self.D))
        self.L.sort_indices()

    def _factor(self, y: np.ndarray) -> np.ndarray:
        r"""
        Matrix-vector product with lower factor, `Z' L^{-T} y`
        Maps from ts.num_edges to ts.num_individuals
        """
        assert self.L.has_sorted_indices
        return self.Z @ self.forward_solve(self.L.indptr, self.L.indices, self.L.data, y)

    def _factor_adjoint(self, y: np.ndarray) -> np.ndarray:
        r"""
        Matrix-vector product with upper factor, `L^{-1} Z y`
        Maps from ts.num_individuals to ts.num_edges
        """
        assert self.L.has_sorted_indices
        return self.backward_solve(self.L.indptr, self.L.indices, self.L.data, self.Z.T @ y)

    def __call__(self, sigma: float, tau: float, y: np.ndarray) -> np.ndarray:
        r"""
        Matrix-vector product, `(Z' (L L')^{-1} Z \tau^2 + I \sigma^2) y`
        """
        is_vector = y.ndim == 1
        if is_vector: y = y.reshape(-1, 1)
        x = y * sigma + self._factor(self._factor_adjoint(y)) * tau
        if is_vector: x = x.squeeze()
        return x

    def solve(
        self, 
        sigma: float, 
        tau: float, 
        y: np.ndarray, 
        preconditioner: NystromPreconditioner = None, 
        maxitt: int = None, 
        atol: float = 0.0, 
        rtol: float = 1e-5,
    ) -> (np.ndarray, int, bool):
        r"""
        Matrix-vector product with inverse, `(Z' (L L')^{-1} Z \tau^2 + I \sigma^2)^{-1} y`

        Re-implemented CG to parallelize with multiple rhs, adapted from:
        https://github.com/scipy/scipy/blob/v1.14.1/scipy/sparse/linalg/_isolve/iterative.py#L283-L388
        """
        is_vector = y.ndim == 1
        assert y.shape[0] == self.dim
        assert atol >= 0 and rtol >= 0
        if maxitt is None: maxitt = self.dim
        if not y.any(): return y, 0, True  # y == 0
        if is_vector: y = y.reshape(-1, 1)
        atol = max(atol, rtol * np.linalg.norm(y))
        M = preconditioner
        x = np.zeros_like(y) if M is None else M(y)
        r = y - self(sigma, tau, x)
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

    def as_matrix(self, sigma: float, tau: float) -> np.ndarray:
        """
        For testing purposes only
        """
        I = np.eye(self.dim)
        Sigma = I * sigma + self._factor(self._factor_adjoint(I)) * tau
        return Sigma

    def simulate(
        self, 
        sigma: float, 
        tau: float, 
        rng: np.random.Generator = None, 
        num_samples: int = 1,
    ) -> np.ndarray:
        """
        Return (edge effects, genetic values, observed values)
        """
        # TODO: should edge effects be scaled by noise
        if rng is None: rng = np.random.default_rng()
        u = rng.normal(size=(self.factor_dim, num_samples)) * np.sqrt(tau)
        g = self._factor(u)
        y = g + rng.normal(size=(self.dim, num_samples)) * np.sqrt(sigma)
        return u.squeeze(), g.squeeze(), y.squeeze(), 


class NystromPreconditioner:
    """
    Randomized Nystrom approximation to the inverse square root of the
    covariance matrix defined by `TraitCovariance`
    """

    @staticmethod
    def _rand_eigh(covariance: TraitCovariance, rank: int, samples: int, seed: int) -> (np.ndarray, np.ndarray):
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

    def __init__(self, covariance: TraitCovariance, rank: int, samples: int = None, seed: int = None):
        samples = rank if samples is None else max(rank, samples)
        self.dim = covariance.dim
        self.D, self.U = self._rand_eigh(covariance, rank, samples, seed)

    def _spectrum(self, sigma: float, tau: float) -> np.ndarray:
        return self.D * tau / sigma + 1

    def __call__(self, sigma: float, tau: float, y: np.ndarray) -> np.ndarray:
        """
        Inverse-vector product: section 17.2 in https://arxiv.org/pdf/2002.01387
        """
        is_vector = y.ndim == 1
        if is_vector: y = y.reshape(-1, 1)
        S = self._spectrum(sigma, tau)
        x = self.U.T @ y * np.expand_dims(1 - 1 / S, 1)
        x = y - self.U @ x
        if is_vector: x = x.squeeze()
        return x

    def as_matrix(self, sigma: float, tau: float) -> np.ndarray:
        """
        Not inverted, for testing only
        """
        S = self._spectrum(sigma, tau)
        return np.eye(self.dim) - self.U @ np.diag(1 - S) @ self.U.T

    def logdet(self, sigma: float, tau: float) -> float:
        """
        Log-determinant of preconditioner
        """
        return np.sum(np.log(self._spectrum(sigma, tau)))

    def gradient(self, sigma: float, tau: float, y: np.ndarray = None) -> float:
        """
        Gradient of log-determinant with regard to spectrum

        Returns `trace(solve(preconditioner, deriv(preconditioner, spectrum)))`,
        or if vectors `y` are passed `preconditioner @ deriv(preconditioner, spectrum) @ y`
        """
        S = self.spectrum(sigma, tau)
        Sd = (1 - 1 / S)
        if y is None:
            return np.sum(Sd)
        assert y.ndim == 2
        Sd = (self.U.T @ y) * Sd[:, np.newaxis]
        return self.U @ Sd
        # isn't this just __call__ minus the identity

    # --- remove these --- #

    def grad_sigma(self, sigma: float, tau: float, y: np.ndarray = None) -> float:
        """
        return trace(solve(preconditioner, deriv(preconditioner, sigma)))
        or if vectors `y` are passed:
        return preconditioner @ deriv(preconditioner, sigma) @ y
        """
        S = self.D * tau / sigma + 1
        grad_sigma = -(1 - 1 / S) / sigma
        if y is None:
            return np.sum(grad_sigma)
        assert y.ndim == 2
        grad_sigma = (self.U.T @ y) * grad_sigma[:, np.newaxis]
        return self.U @ grad_sigma

    def grad_tau(self, sigma: float, tau: float, y: np.ndarray = None) -> float:
        """
        Gradient of log-determinant with regard to tau.

        Returns `trace(solve(preconditioner, deriv(preconditioner, tau)))`,
        or if vectors `y` are passed `preconditioner @ deriv(preconditioner, tau) @ y`
        """
        S = self.D * tau / sigma + 1
        grad_tau = (1 - 1 / S) / tau
        if y is None:
            return np.sum(grad_tau)
        assert y.ndim == 2
        grad_tau = (self.U.T @ y) * grad_tau[:, np.newaxis]
        return self.U @ grad_tau


