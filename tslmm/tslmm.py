from __future__ import annotations

import numpy as np
import scipy.sparse
import numba
from typing import Callable

from tslmm.trace_estimators import xtrace, hutchinson, xdiag
from tslmm.operations import split_upwards
from tslmm.matrices import edge_individual_matrix, edge_adjacency


_i1r = numba.types.Array(numba.types.int32, 1, 'C', readonly=True)
_f1r = numba.types.Array(numba.types.float64, 1, 'C', readonly=True)
_f2r = numba.types.Array(numba.types.float64, 2, 'C', readonly=True)
_f2w = numba.types.Array(numba.types.float64, 2, 'C', readonly=False)


# --- for testing --- #

def _explicit_covariance_matrix(sigma, tau, tree_sequence, mutation_rate, center_around=None):
        ts = split_upwards(tree_sequence)
        D = mutation_rate * (ts.edges_right - ts.edges_left) * \
            (ts.nodes_time[ts.edges_parent] - ts.nodes_time[ts.edges_child])
        Z = edge_individual_matrix(ts).T.toarray()
        L = scipy.sparse.identity(ts.num_edges) - edge_adjacency(ts).T
        L = L.T @ scipy.sparse.diags_array(1 / np.sqrt(D))
        Q = L @ L.T
        I = scipy.sparse.eye_array(ts.num_individuals, format='csr')
        G = scipy.sparse.linalg.spsolve_triangular(L, Z.T)
        if center_around is not None: 
            G - np.mean(G[:, center_around], axis=1)[:, np.newaxis]
        return I * sigma + G.T @ G * tau


def _explicit_reml(sigma, tau, tree_sequence, mutation_rate, y, X, subset=None, center_covariance=False):
    if subset is None: subset = np.arange(tree_sequence.num_individuals)
    G = _explicit_covariance_matrix(
        sigma, tau, tree_sequence, mutation_rate, 
        center_around=subset if center_covariance else None,
    )
    G, X, y = G[np.ix_(subset, subset)], X[subset], y[subset]
    sign, Gldet = np.linalg.slogdet(G)
    assert sign > 0
    GinvX = np.linalg.solve(G, X)
    GinvY = np.linalg.solve(G, y)
    XinvX = X.T @ GinvX
    resid = y - X @ np.linalg.solve(XinvX, X.T @ GinvY)
    Ginvr = np.linalg.solve(G, resid)  
    sign, Xldet = np.linalg.slogdet(XinvX)
    assert sign > 0
    deviance = Gldet + Xldet + resid.T @ Ginvr
    return -deviance / 2


def _explicit_gradient(sigma, tau, tree_sequence, mutation_rate, y, X, subset=None, center_covariance=False):
    if subset is None: subset = np.arange(tree_sequence.num_individuals)
    G = _explicit_covariance_matrix(
        sigma, tau, tree_sequence, mutation_rate,
        center_around=subset if center_covariance else None,
    )
    G, X, y = G[np.ix_(subset, subset)], X[subset], y[subset]
    dGdt = (G - np.eye(y.size) * sigma) / tau
    dGds = np.eye(y.size)
    GinvX = np.linalg.solve(G, X)
    GinvY = np.linalg.solve(G, y)
    XinvX = X.T @ GinvX
    resid = y - X @ np.linalg.solve(XinvX, X.T @ GinvY)
    Ginvr = np.linalg.solve(G, resid)
    dy_ds = X @ np.linalg.solve(XinvX, GinvX.T @ (dGds @ Ginvr))
    dy_dt = X @ np.linalg.solve(XinvX, GinvX.T @ (dGdt @ Ginvr))
    grad_s = np.sum(np.linalg.inv(G).diagonal()) - \
        np.sum(np.linalg.solve(XinvX, GinvX.T @ (dGds @ GinvX)).diagonal()) - \
        np.dot(Ginvr.T, dGds @ Ginvr) + 2 * dy_ds.T @ Ginvr
    grad_t = np.sum(np.linalg.solve(G, dGdt).diagonal()) - \
        np.sum(np.linalg.solve(XinvX, GinvX.T @ (dGdt @ GinvX)).diagonal()) - \
        np.dot(Ginvr.T, dGdt @ Ginvr) + 2 * dy_dt.T @ Ginvr
    return -grad_s / 2, -grad_t / 2


def _explicit_posterior(sigma, tau, tree_sequence, mutation_rate, y, X, subset=None, predict_subset=None, center_covariance=False):
    if subset is None: subset = np.arange(tree_sequence.num_individuals)
    if predict_subset is None: predict_subset = subset
    GRM = _explicit_covariance_matrix(
        0, tau, tree_sequence, mutation_rate, 
        center_around=subset if center_covariance else None,
    )
    i, j = subset, predict_subset
    Gii = GRM[np.ix_(i, i)]
    X, y = X[i], y[i]
    Sigma = Gii + np.eye(i.size) * sigma
    varcov = X.T @ np.linalg.solve(Sigma, X)
    fixeff = np.linalg.solve(varcov, X.T @ np.linalg.solve(Sigma, y))
    resids = y - X @ fixeff
    scaled = np.linalg.solve(Sigma, resids)
    Gij = GRM[np.ix_(i, j)]
    Gjj = GRM[np.ix_(j, j)]
    posterior_mean = Gij.T @ scaled
    posterior_vcov = Gjj - Gij.T @ np.linalg.solve(Sigma, Gij)
    return posterior_mean, posterior_vcov


# --- linear operators --- #

def genetic_relatedness_vector(
        ts: tskit.Treesequence,
        mat: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        centre: bool = False,
        windows = None,
        ):
    
    # maps samples to individuals
    def sample_individual_sparray(ts: tskit.TreeSequence) -> scipy.sparse.sparray:
        samples_individual = ts.nodes_individual[ts.samples()]
        return scipy.sparse.csr_array(
                (
                    np.ones(ts.num_samples),
                    (np.arange(ts.num_samples), samples_individual)
                ),
                shape=(ts.num_samples, ts.num_individuals)
            )
    
    # maps values in idx to num_individuals
    def individual_idx_sparray(n: int, idx: np.ndarray) -> scipy.sparse.sparray:
        return scipy.sparse.csr_array(
                (
                    np.ones(idx.size),
                    (idx, np.arange(idx.size))
                ),
                shape=(n, idx.size)
            )
                        
    assert cols.size == mat.shape[0], "Dimension mismatch"
    # centering
    x = mat - mat.mean(axis=0) if centre else mat
    x = individual_idx_sparray(ts.num_individuals, cols).dot(x)
    x = sample_individual_sparray(ts).dot(x)
    x = ts.genetic_relatedness_vector(W=x, windows=windows, mode="branch", centre=False)
    x = sample_individual_sparray(ts).T.dot(x)
    x = individual_idx_sparray(ts.num_individuals, rows).T.dot(x)
    x = x - x.mean(axis=0) if centre else x

    return x

class CovarianceModel:
    """
    Covariance between phenotypes `y = Z u + e` 
    where `u ~ N(0, tau L^{-1} L^{-T})` and `e ~ N(0, sigma I)`. 
    Hence, `y ~ N(0, tau Z L^{-1} L^{-T} Z^T + sigma I)`
    """
    """
    The class either does matvec or solve
    """
   
    def __init__(self, ts: tskit.TreeSequence, mutation_rate: float = 1.0):
        # TODO: mean centering around a subset
        """
        ts = split_upwards(tree_sequence)
        self.dim = ts.num_individuals
        self.factor_dim = ts.num_edges
        self.D = mutation_rate * (ts.edges_right - ts.edges_left) * \
            (ts.nodes_time[ts.edges_parent] - ts.nodes_time[ts.edges_child])
        self.Z = edge_individual_matrix(ts).T.tocsr()
        self.L = scipy.sparse.identity(ts.num_edges) - edge_adjacency(ts).T
        self.L = self.L.T @ scipy.sparse.diags_array(1 / np.sqrt(self.D))
        self.L.sort_indices()
        self.I = scipy.sparse.eye_array(ts.num_individuals, format='csr')
        """
        self.dim = ts.num_individuals
        self.I_indices = np.arange(self.dim)
        self.mutation_rate = mutation_rate
        self.ts = ts
        self.I = scipy.sparse.eye_array(ts.num_individuals, format='csr')

    def __call__(
        self, sigma: float, tau: float, y: np.ndarray, 
        rows: np.ndarray = None, cols: np.ndarray = None,
        centre: bool = False,
    ) -> np.ndarray:
        r"""
        Matrix-vector product, `(Z (L L')^{-1} Z' \tau^2 + I \sigma^2) y`, optionally
        with the submatrix specified by `rows` and `cols`
        """
        is_vector = y.ndim == 1
        if rows is None: rows = self.I_indices
        if cols is None: cols = self.I_indices
        if is_vector: y = y.reshape(-1, 1)
        assert cols.size == y.shape[0], "Dimension mismatch"
        
        x = genetic_relatedness_vector(self.ts, y, rows, cols, centre) 
        x *= self.mutation_rate
        x = tau * x + sigma * self.I[rows] @ (self.I[cols].T @ y)
        if is_vector: x = x.squeeze()
        return x

    def solve(
        self, 
        sigma: float, 
        tau: float, 
        y: np.ndarray, 
        preconditioner: LowRankPreconditioner = None, 
        indices: np.ndarray = None,
        maxitt: int = None, 
        atol: float = 0.0, 
        rtol: float = 1e-5,
        return_info: bool = False,
    ) -> (np.ndarray, int, bool):
        r"""
        Matrix-vector product with inverse, `(Z' (L L')^{-1} Z \tau^2 + I \sigma^2)^{-1} y`,
        optionally inverts the submatrix indexed by `indices`
        
        Re-implemented CG to parallelize with multiple rhs, adapted from:
        https://github.com/scipy/scipy/blob/v1.14.1/scipy/sparse/linalg/_isolve/iterative.py#L283-L388
        """
        is_vector = y.ndim == 1
        dim = self.dim if indices is None else indices.size
        assert y.shape[0] == dim
        assert atol >= 0 and rtol >= 0
        if maxitt is None: maxitt = dim
        if not y.any(): return y, 0, True  # y == 0
        if is_vector: y = y.reshape(-1, 1)
        atol = max(atol, rtol * np.linalg.norm(y))
        A = lambda x: self(sigma, tau, x, rows=indices, cols=indices)
        M = preconditioner
        x = np.zeros_like(y)
        r = y - A(x)
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
            q = A(p)
            alpha = rho / np.sum(p * q, axis=0)
            x += alpha[np.newaxis] * p
            r -= alpha[np.newaxis] * q
            last_rho = rho
        success = itt + 1 < maxitt
        if not success: print("CG did not converge") # DEBUG
        if is_vector: x = x.squeeze()
        return (x, (itt, success)) if return_info else x
     
    def expand_to_dim(self, x: np.ndarray, ind: np.ndarray) -> np.ndarray:
        assert ind.size == x.shape[0]
        data = np.ones(ind.size)
        row_ind, col_ind = ind, np.arange(ind.size) 
        return scipy.sparse.csr_array(
            (data, (row_ind, col_ind)),
            shape=(self.dim, ind.size)
        ).dot(x)

class LowRankPreconditioner:
    """
    Randomized approximation to the inverse of `CovarianceModel` with scaled
    eigenvalues
    """

    @staticmethod
    def _rand_eigh(
        operator: Callable,
        operator_dim: int,
        rank: int, 
        num_vectors: int, 
        rng: np.random.Generator,
    ) -> (np.ndarray, np.ndarray):
        """
        Algorithm 16 in https://arxiv.org/pdf/2002.01387
        """
        assert num_vectors >= rank > 0
        test_vectors = rng.normal(size=(operator_dim, num_vectors))
        test_vectors = np.linalg.qr(test_vectors).Q  # orthonormalise
        proj_vectors = operator(test_vectors)
        shift = np.sqrt(operator_dim) * np.spacing(np.linalg.norm(proj_vectors))
        proj_vectors += shift * test_vectors
        chol_factor = np.linalg.cholesky(test_vectors.T @ proj_vectors)
        coef_matrix = np.linalg.solve(chol_factor, proj_vectors.T)
        U, D, _ = np.linalg.svd(coef_matrix.T, full_matrices=False)
        D = np.maximum(D ** 2 - shift, 0)
        return D[:rank], U[:, :rank]

    def __init__(
        self, 
        covariance: CovarianceModel, 
        rank: int, 
        indices: np.ndarray = None, 
        num_vectors: int = None, 
        rng: np.random.Generator = None,
    ):
        if rng is None: rng = np.random.default_rng()
        num_vectors = rank if num_vectors is None else max(rank, num_vectors)
        self.dim = covariance.dim if indices is None else indices.size
        self.D, self.U = self._rand_eigh(
            lambda x: covariance(0, 1, x, rows=indices, cols=indices), 
            operator_dim=self.dim, 
            rank=rank, 
            num_vectors=num_vectors, 
            rng=rng,
        )

    def __call__(self, sigma: float, tau: float, y: np.ndarray) -> np.ndarray:
        """
        Inverse-vector product: section 17.2 in https://arxiv.org/pdf/2002.01387
        """
        is_vector = y.ndim == 1
        if is_vector: y = y.reshape(-1, 1)
        S = self.D * tau / sigma + 1
        x = self.U.T @ y * np.expand_dims(1 - 1 / S, 1)
        x = y - self.U @ x
        if is_vector: x = x.squeeze()
        return x


class tslmm:

    @staticmethod
    def _reml_stochastic_gradient(
        sigma: float, 
        tau: float, 
        phenotypes: np.ndarray, 
        covariates: np.ndarray, 
        covariance: CovarianceModel, 
        preconditioner: LowRankPreconditioner, 
        indices: np.ndarray = None,
        trace_samples: int = 1, 
        rng: np.random.Generator = None,
    ):
        """
        Unbiased estimate of gradient of `REML(sigma, tau | traits, covariates, tree_sequence)`
        Also returns fixed effects and residuals
        """
        assert trace_samples > 0
        assert min(sigma, tau) > 0

        # TODO: this is written in a way so that it's easy to generalize, but sigma can be factored out so that
        # we only need one trace estimate

        def _dGds(test_vectors):  # d(covariance)/d(sigma)
            return test_vectors

        def _dGdt(test_vectors):  # d(covariance)/d(tau)
            return covariance(0, 1, test_vectors, rows=indices, cols=indices)

        def _Ginv_dGds_trace(test_vectors):
            return covariance.solve(sigma, tau, _dGds(test_vectors), preconditioner=M, indices=indices)

        def _Ginv_dGdt_trace(test_vectors):
            return covariance.solve(sigma, tau, _dGdt(test_vectors), preconditioner=M, indices=indices)

        if rng is None: rng = np.random.default_rng()
        # TODO: trace_samples == 1 is erroring out
        trace_estimator = hutchinson if trace_samples == 1 else xtrace
        M = None if preconditioner is None else lambda x: preconditioner(sigma, tau, x)
        dim = covariance.dim if indices is None else indices.size

        y, X = phenotypes, covariates
        Xy = np.hstack([X, y.reshape(-1, 1)])
        Ginv_Xy = covariance.solve(sigma, tau, Xy, preconditioner=M, indices=indices)
        Xt_Ginv, Ginv_y = Ginv_Xy.T[:-1], Ginv_Xy.T[-1]
        Xt_Ginv_X_inv = np.linalg.inv(Xt_Ginv @ X)
        fixed_effects = Xt_Ginv_X_inv @ X.T @ Ginv_y
        residuals = y - X @ fixed_effects
        Ginv_r = covariance.solve(sigma, tau, residuals, preconditioner=M, indices=indices)

        dGds_Ginv_r = _dGds(Ginv_r)
        dGds_Ginv_X = _dGds(Xt_Ginv.T)
        Xt_Ginv_dGds_Ginv_r = Xt_Ginv @ dGds_Ginv_r
        Xt_Ginv_dGds_Ginv_X = Xt_Ginv @ dGds_Ginv_X
        Ginv_dGds_trace, _ = trace_estimator(_Ginv_dGds_trace, dim, trace_samples, rng=rng)
        dy_ds = X @ (Xt_Ginv_X_inv @ Xt_Ginv_dGds_Ginv_r)  # TODO: this partial should be zero?
        sigma_grad = -np.sum(Xt_Ginv_X_inv * Xt_Ginv_dGds_Ginv_X) - \
            np.sum(Ginv_r * (dGds_Ginv_r - 2 * dy_ds)) + Ginv_dGds_trace

        dGdt_Ginv_r = _dGdt(Ginv_r)
        dGdt_Ginv_X = _dGdt(Xt_Ginv.T)
        Xt_Ginv_dGdt_Ginv_r = Xt_Ginv @ dGdt_Ginv_r
        Xt_Ginv_dGdt_Ginv_X = Xt_Ginv @ dGdt_Ginv_X
        Ginv_dGdt_trace, _ = trace_estimator(_Ginv_dGdt_trace, dim, trace_samples, rng=rng)
        dy_dt = X @ (Xt_Ginv_X_inv @ Xt_Ginv_dGdt_Ginv_r)  # TODO: this partial should be zero?
        tau_grad = -np.sum(Xt_Ginv_X_inv * Xt_Ginv_dGdt_Ginv_X) - \
            np.sum(Ginv_r * (dGdt_Ginv_r - 2 * dy_dt)) + Ginv_dGdt_trace

        gradient = np.array([-sigma_grad / 2, -tau_grad / 2])
        return gradient, fixed_effects, residuals


    @staticmethod
    def _reml_stochastic_optimize(
        starting_values: np.ndarray,
        phenotypes: np.ndarray, 
        covariates: np.ndarray, 
        covariance: CovarianceModel,
        preconditioner: LowRankPreconditioner,
        indices: np.ndarray = None,
        trace_samples: int = 10, 
        decay: float = 0.1, 
        epsilon: float = 1e-4, 
        min_value: float = 1e-4,
        max_iterations: int = 100,
        verbose: bool = True,
        callback: Callable = None,
        rng: np.random.Generator = None, 
    ):
        """
        Set `max_iterations` to zero to skip optimization
        """
        if rng is None: rng = np.random.default_rng()

        # scale things so that hyperparameters are easier to default
        mean, scale = np.mean(phenotypes), np.std(phenotypes)
        y = (phenotypes - mean) / scale
        X, R = np.linalg.qr(covariates)

        # TODO: use better starting values, like HE regression
        if starting_values is None:
            ols = covariates @ np.linalg.solve(R.T @ R, covariates.T @ y)
            ols = np.var(y - ols, ddof=covariates.shape[1])
            starting_values = np.full(2, np.sqrt(ols / 2))
        else:
            starting_values = np.sqrt(starting_values) / scale
        # AdaDelta (https://arxiv.org/pdf/1212.5701)
        state = starting_values
        running_mean = state
        numerator = np.zeros(state.size)
        denominator = np.zeros(state.size)
        for itt in range(max_iterations):
            state = np.clip(state, min_value, np.inf)  # force positive
            gradient, _, _ = tslmm._reml_stochastic_gradient(
                *np.power(state, 2), y, X, covariance, preconditioner, 
                trace_samples=trace_samples, indices=indices, rng=rng,
            )
            gradient *= 2 * state  # variance to std deviation
            # state += epsilon * gradient  # usual SGD update
            denominator = (1 - decay) * denominator + decay * gradient ** 2
            update = np.sqrt(numerator + epsilon) / np.sqrt(denominator + epsilon) * gradient
            numerator = (1 - decay) * numerator + decay * update ** 2
            state = state + update
            running_mean = (1 - decay) * running_mean + decay * state
            if verbose: print(f"Iteration {itt}: {np.power(scale * running_mean, 2).round(2)}")
            if callback is not None: callback(np.power(scale * running_mean, 2))
            # TODO: stopping condition based on change in running mean
            # TODO: fit a quadratic using gradient from last K iterations to get better estimate

        _, fixed_effects, residuals = tslmm._reml_stochastic_gradient(
            *np.power(running_mean, 2), y, X, covariance, preconditioner, 
            trace_samples=trace_samples, indices=indices, rng=rng
        )
        fixed_effects = np.linalg.solve(R, fixed_effects) * scale
        running_mean *= scale
        residuals *= scale
        residuals += mean

        return np.power(running_mean, 2), fixed_effects, residuals

    @staticmethod
    def _haseman_elston_regression(
        phenotypes: np.ndarray,
        covariates: np.ndarray,
        covariance: CovarianceModel,
        indices: np.ndarray = None,
        trace_samples: int = 10,
        rng: np.random.Generator = None,
    ):
        """
        Haseman-Elston regression for REML initialization
        """
        assert trace_samples > 0
        
        if rng is None: rng = np.random.default_rng()
        trace_estimator = hutchinson if trace_samples == 1 else xtrace
        dim = covariance.dim if indices is None else indices.size

        # scale things so that hyperparameters are easier to default
        y = phenotypes - np.mean(phenotypes)
        X, R = np.linalg.qr(covariates)

        def _projection(test_vectors):
            return test_vectors - X @ (X.T @ test_vectors)

        def _G(test_vectors):
            return covariance(0, 1, test_vectors, rows=indices, cols=indices)

        def _PG(test_vectors):
            return _projection(covariance(0, 1, test_vectors, rows=indices, cols=indices))

        def _PGPG(test_vectors):
            return _PG(_PG(test_vectors))

        # tr(PGPG), tr(PG), tr(P)
        PGPG_trace, _ = trace_estimator(_PGPG, dim, trace_samples,rng=rng)
        PG_trace, _ = trace_estimator(_PG, dim, trace_samples, rng=rng)
        P_trace = dim - covariates.shape[1] # N-P
        
        # yPGPy, yPy 
        Py = _projection(y)
        yPy = np.dot(y, Py)
        GPy = _G(Py)
        yPGPy = np.dot(Py, GPy)

        # solve
        a = np.array([
                [PGPG_trace, PG_trace],
                [PG_trace, P_trace]
             ])
        b = np.array([yPGPy, yPy])
        state = np.linalg.solve(a, b)
    
        return state[::-1]

    # ------ API ------ #

    def __init__(
        self, 
        tree_sequence: tskit.TreeSequence, 
        mutation_rate: float,
        phenotypes: np.ndarray, 
        covariates: np.ndarray = None, 
        phenotyped_individuals: np.ndarray = None,
        variance_components: np.ndarray = None,
        sgd_iterations: int = 50,  # TODO: use a stopping criterion
        sgd_decay: float = 0.1,
        sgd_epsilon: float = 1e-4,  # if this is too large SGD will diverge
        sgd_samples: int = 5,
        sgd_verbose: bool = True,
        preconditioner_rank: int = 10,
        rng: np.random.Generator = None,
        initialization: str = None,
    ):
        """
        NB: `variance_components` are assumed fixed if provided; otherwise stochastic
        gradient descent is used to fit the model via REML.

        NB: `covariates` should not contain an intercept, as this is unidentifiable
        with the current GRM
        """

        if rng is None: rng = np.random.default_rng()

        if phenotyped_individuals is None:
            phenotyped_individuals = np.arange(tree_sequence.num_individuals)

        if covariates is None:  # TODO but not identifiable w/ intercept
            covariates = np.ones((phenotyped_individuals.size, 1))

        assert phenotyped_individuals.size == phenotypes.size == covariates.shape[0]

        self.phenotyped_individuals = phenotyped_individuals
        self.covariates = covariates
        self.phenotypes = phenotypes
        self.covariance = CovarianceModel(tree_sequence, mutation_rate=mutation_rate)
        self.preconditioner = None if preconditioner_rank < 1 else \
            LowRankPreconditioner(
                self.covariance, 
                rank=preconditioner_rank, 
                num_vectors=2 * preconditioner_rank, 
                indices=self.phenotyped_individuals,
                rng=rng,
            )
        if initialization == "he":
            variance_components = self._haseman_elston_regression(
                phenotypes=self.phenotypes,
                covariates=self.covariates,
                covariance=self.covariance,
                indices=self.phenotyped_individuals,
                trace_samples=sgd_samples,
                rng=rng,
            )
            print(variance_components)

        self._optimization_trajectory = []
        self.variance_components, self.fixed_effects, self.residuals = \
            self._reml_stochastic_optimize(
                starting_values=variance_components,
                phenotypes=self.phenotypes,
                covariates=self.covariates,
                covariance=self.covariance,
                preconditioner=self.preconditioner,
                indices=self.phenotyped_individuals,
                max_iterations=sgd_iterations, # if variance_components is None else 0,
                trace_samples=sgd_samples,
                epsilon=sgd_epsilon,
                decay=sgd_decay,
                verbose=sgd_verbose,
                rng=rng,
                callback=lambda x: self._optimization_trajectory.append(x),
            )

    def predict(self, individuals: np.ndarray = None, variance_samples: int = 0, rng: np.random.Generator = None):
        """
        Return the posterior mean genetic values (BLUPs) for _all_ individuals
        in the tree sequence, and (if `variance_samples` is nonzero) a Monte
        Carlo estimate of the posterior variance
        """

        if rng is None: rng = np.random.default_rng()
        if individuals is None: individuals = np.arange(self.covariance.dim)
        i, j = self.phenotyped_individuals, individuals
        sigma, tau = self.variance_components
        M = lambda x: self.preconditioner(sigma, tau, x)

        def _posterior_var(test_vectors):
            sketch = self.covariance(0, tau, test_vectors, rows=i, cols=j)
            sketch = self.covariance.solve(sigma, tau, sketch, preconditioner=M, indices=i)
            sketch = self.covariance(0, tau, sketch, rows=j, cols=i)
            sketch = self.covariance(0, tau, test_vectors, rows=j, cols=j) - sketch
            return sketch

        # TODO do the solve as part of the optimization routine
        self.weighted_residuals = self.covariance.solve(sigma, tau, self.residuals, preconditioner=M, indices=i)
        E_g = self.covariance(0, tau, self.weighted_residuals, rows=j, cols=i)
        if variance_samples > 0: V_g = xdiag(_posterior_var, j.size, variance_samples, rng) 

        return (E_g, V_g) if variance_samples > 0 else E_g
    


