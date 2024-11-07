from __future__ import annotations

import numpy as np
import scipy.sparse
import numba
from typing import Callable

from tslmm.trace_estimators import xtrace, hutchinson, xdiag
from tslmm.tspca import _rand_svd
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


def _explicit_average_information(sigma, tau, tree_sequence, mutation_rate, y, X, subset=None, center_covariance=False):
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
    Xt_Ginv_X = X.T @ GinvX
    resid = y - X @ np.linalg.solve(Xt_Ginv_X, X.T @ GinvY)
    Ginvr = np.linalg.solve(G, resid)
    
    dGds_Ginvr = dGds @ Ginvr
    dGdt_Ginvr = dGdt @ Ginvr
    
    Ginv_dGds_Ginvr = np.linalg.solve(G, dGds_Ginvr)
    Ginv_dGdt_Ginvr = np.linalg.solve(G, dGdt_Ginvr)
    Xt_Ginv_dGds_Ginvr = X.T @ Ginv_dGds_Ginvr
    Xt_Ginv_dGdt_Ginvr = X.T @ Ginv_dGdt_Ginvr
    Xt_Ginv_X_inv = np.linalg.inv(Xt_Ginv_X)

    I_ss = np.dot(Ginv_dGds_Ginvr, dGds_Ginvr) - Xt_Ginv_dGds_Ginvr.T @ Xt_Ginv_X_inv @ Xt_Ginv_dGds_Ginvr
    I_st = np.dot(Ginv_dGds_Ginvr, dGdt_Ginvr) - Xt_Ginv_dGds_Ginvr.T @ Xt_Ginv_X_inv @ Xt_Ginv_dGdt_Ginvr
    I_tt = np.dot(Ginv_dGdt_Ginvr, dGdt_Ginvr) - Xt_Ginv_dGdt_Ginvr.T @ Xt_Ginv_X_inv @ Xt_Ginv_dGdt_Ginvr
    average_information = np.array([[I_ss, I_st], [I_st, I_tt]])

    return average_information


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
        arr: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        centre: bool = False,
        windows = None,
        ) -> np.ndarray:
    """
    Wrapper around `tskit.TreeSequence.genetic_relatedness_vector` to support centering in respect to individuals.
    Multiplies an array to the genetic relatedness matrix of :class:`tskit.TreeSequence`.

    :param tskit.TreeSequence ts: A tree sequence.
    :param numpy.ndarray arr: The array to multiply. Either a vector or a matrix.
    :param numpy.ndarray rows: Index of rows of the genetic relatedness matrix to be selected.
    :param numpy.ndarray cols: Index of cols of the genetic relatedness matrix to be selected. The size should match the row length of `arr`.
    :param bool centre: Centre the genetic relatedness matrix. Centering happens respect to the `rows` and `cols`. 
    :param windows: An increasing list of breakpoints between the windows to compute the genetic relatedness matrix in.
    :return: An array that is the matrix-array product of the genetic relatedness matrix and the array. 
    :rtype: `np.ndarray`
    """
    
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
                        
    assert cols.size == arr.shape[0], "Dimension mismatch"
    # centering
    x = arr - arr.mean(axis=0) if centre else arr # centering within index in rows
    x = individual_idx_sparray(ts.num_individuals, cols).dot(x)
    x = sample_individual_sparray(ts).dot(x)
    x = ts.genetic_relatedness_vector(W=x, windows=windows, mode="branch", centre=False)
    x = sample_individual_sparray(ts).T.dot(x)
    x = individual_idx_sparray(ts.num_individuals, rows).T.dot(x)
    x = x - x.mean(axis=0) if centre else x # centering within index in cols

    return x

class CovarianceModel:
    """
    The covariance matrix between phenotypes `y = Z u + e`
    where `u ~ N(0, tau G) and e ~ N(0, sigma I)`.
    The matrix is defined implicitly as a matrix (or its inverse) - array product.
    The inverse is computed by conjugate gradient.

    :param tskit.TreeSequence ts: A tree sequence.
    :param float mutation_rate: Mutation rate. 
    """
       
    def __init__(self, ts: tskit.TreeSequence, mutation_rate: float = 1.0):
        # TODO: mean centering around a subset
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
        Multiplies the covariance matrix to an array.

        :param float sigma: Parameter in `sigma I`. It is proportional to the non-genetic variance.
        :param float tau: Parameter in `tau G`. It is proportional to the genetic (mutational) variance.
        :param np.ndarray y: Array to be multiplied to the GRM from the right.
        :param numpy.ndarray rows: Index of rows of the genetic relatedness matrix to be selected.
        :param numpy.ndarray cols: Index of cols of the genetic relatedness matrix to be selected. The size should match the row length of `arr`.
        :param bool centre: Centre the genetic relatedness matrix. Centering happens respect to the `rows` and `cols`. 
        :return: Product of the covariance matrix and the array.
        :rtype: np.ndarray
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
        Multiplies the inverse of the covariance matrix to an array.
        The conjugate-gradient algorithm was adopted from:
        https://github.com/scipy/scipy/blob/v1.14.1/scipy/sparse/linalg/_isolve/iterative.py#L283-L388

        :param float sigma: Parameter in `sigma I`. It is proportional to the non-genetic variance.
        :param float tau: Parameter in `tau G`. It is proportional to the genetic (mutational) variance.
        :param np.ndarray y: Array to be multiplied to the GRM from the right.
        :param preconditioner: A Low-rank matrix that approximates the covariance matrix. Randomized Nystrome low-rank preconditioner is used by default.
        :param indices: The row and the col indices to subset from the covariance matrix.
        :return: Product of the inverse of the covariance matrix and the array.
        :rtype: np.ndarray
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
     

class LowRankPreconditioner:
    """
    Randomized approximation to the inverse of `CovarianceModel` with scaled
    eigenvalues

    :param Callable covariance: The covariance matrix in which the preconditioner approximates.
    :param int rank: The rank of the preconditioner.
    :param int depth: The number of power iterations used in range finder.
    :param np.ndarray indices: The row and the col indices to subset from the covariance matrix.
    :param int num_vectors: The number of test vectors to use. It should be larger than the rank.
    :param np.random.Generator rng: Random number generator.
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
        assert num_vectors >= rank > 0, "More test vecotrs needed than rank"
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
        depth: int = 1,
        indices: np.ndarray = None, 
        num_vectors: int = None, 
        rng: np.random.Generator = None,
    ):
        if rng is None: rng = np.random.default_rng()
        num_vectors = rank if num_vectors is None else max(rank, num_vectors)
        self.dim = covariance.dim if indices is None else indices.size
        if depth > 1:
            self.U, self.D, _ = _rand_svd(
                lambda x: covariance(0, 1, x, rows=indices, cols=indices),
                operator_dim=self.dim,
                rank=rank,
                depth=depth,
                num_vectors=num_vectors,
                rng=rng,
            )
        else:
            self.D, self.U = self._rand_eigh(
                lambda x: covariance(0, 1, x, rows=indices, cols=indices), 
                operator_dim=self.dim, 
                rank=rank, 
                num_vectors=num_vectors, 
                rng=rng,
            )

    def __call__(self, sigma: float, tau: float, y: np.ndarray) -> np.ndarray:
        """
        The product of the preconditioner, which is a low-rank approximate inverse,
        and the vector.
        See section 17.2 in https://arxiv.org/pdf/2002.01387.

        :param float sigma: Parameter in `sigma I`. It is proportional to the non-genetic variance.
        :param float tau: Parameter in `tau G`. It is proportional to the genetic (mutational) variance.
        :param np.ndarray y: Array to be multiplied to the preconditioner from the right.
        """
        is_vector = y.ndim == 1
        if is_vector: y = y.reshape(-1, 1)
        S = self.D * tau / sigma + 1
        x = self.U.T @ y * np.expand_dims(1 - 1 / S, 1)
        x = y - self.U @ x
        if is_vector: x = x.squeeze()
        return x


class tslmm:
    """
    A tslmm instance to fit ARG-LMM with restricted maximum likelihood (REML).
    The internal algorithms are based on RandNLA (Randomized Numerical Linear Algebra)
    and the incremental algorithm on tree sequences.

    :param tskit.Treesequence ts: A tree sequence.
    :param float mutation_rate: Mutation rate.
    :param np.ndarray phenotypes: Array storing phenotypes.
    :param np.ndarray covariates: Fixed effects covariates.
    :param phenotyped_individuals: Index of phenotyped individuals. Used for prediction of unobserved individiuals.
    :param int preconditioner_rank: The rank of the preconditioner.
    :param int preconditioner_depth: Number of power iterations to make the preconditioner.
    :param rng: Random number generator
    """

    @staticmethod
    def _reml_stochastic_average_information(
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
        Average information is computed along
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

        # gradient incredients
        Xt_Ginv_dGds_Ginv_X = Xt_Ginv @ _dGds(Xt_Ginv.T)
        Xt_Ginv_dGdt_Ginv_X = Xt_Ginv @ _dGdt(Xt_Ginv.T)
        dGds_Ginv_r = _dGds(Ginv_r)
        dGdt_Ginv_r = _dGdt(Ginv_r)

        # d(sigma)
        Ginv_dGds_trace, _ = trace_estimator(_Ginv_dGds_trace, dim, trace_samples, rng=rng)
        ytP_dGds_Py = np.dot(dGds_Ginv_r, Ginv_r) # Ginv_r = Py
        sigma_grad = Ginv_dGds_trace - ytP_dGds_Py - (Xt_Ginv_dGds_Ginv_X * Xt_Ginv_X_inv).sum()

        # d(tau)
        Ginv_dGdt_trace, _ = trace_estimator(_Ginv_dGdt_trace, dim, trace_samples, rng=rng)
        ytP_dGdt_Py = np.dot(dGdt_Ginv_r, Ginv_r) # Ginv_r = Py
        tau_grad = Ginv_dGdt_trace - ytP_dGdt_Py - (Xt_Ginv_dGdt_Ginv_X * Xt_Ginv_X_inv).sum()

        # gradient
        gradient = np.array([-sigma_grad / 2, -tau_grad / 2])

        # information ingredients
        Ginv_dGds_Ginv_r = covariance.solve(sigma, tau, dGds_Ginv_r, preconditioner=M, indices=indices)
        Ginv_dGdt_Ginv_r = covariance.solve(sigma, tau, dGdt_Ginv_r, preconditioner=M, indices=indices)
        Xt_Ginv_dGds_Ginv_r = X.T @ Ginv_dGds_Ginv_r
        Xt_Ginv_dGdt_Ginv_r = X.T @ Ginv_dGdt_Ginv_r

        # average information
        I_ss = np.dot(Ginv_dGds_Ginv_r, dGds_Ginv_r) - Xt_Ginv_dGds_Ginv_r.T @ Xt_Ginv_X_inv @ Xt_Ginv_dGds_Ginv_r
        I_st = np.dot(Ginv_dGds_Ginv_r, dGdt_Ginv_r) - Xt_Ginv_dGds_Ginv_r.T @ Xt_Ginv_X_inv @ Xt_Ginv_dGdt_Ginv_r
        I_tt = np.dot(Ginv_dGdt_Ginv_r, dGdt_Ginv_r) - Xt_Ginv_dGdt_Ginv_r.T @ Xt_Ginv_X_inv @ Xt_Ginv_dGdt_Ginv_r
        average_information = np.array([[I_ss, I_st], [I_st, I_tt]])

        return gradient, fixed_effects, residuals, average_information


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
            starting_values = np.clip(starting_values, min_value, np.inf)
            starting_values = np.sqrt(starting_values) / scale
        # AdaDelta (https://arxiv.org/pdf/1212.5701)
        state = starting_values
        running_mean = state
        numerator = np.zeros(state.size)
        denominator = np.zeros(state.size)
        if callback is not None: callback(np.power(scale * running_mean, 2))
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
            if verbose: print(f"Iteration {itt}: {np.power(scale * running_mean, 2).round(2)}, {np.linalg.norm(gradient)}")
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
    def _reml_stochastic_optimize_ai(
        starting_values: np.ndarray,
        phenotypes: np.ndarray, 
        covariates: np.ndarray, 
        covariance: CovarianceModel,
        preconditioner: LowRankPreconditioner,
        indices: np.ndarray = None,
        trace_samples: int = 10, 
        min_value: float = 1e-4,
        max_iterations: int = 100,
        stop_rtol: int = 5e-2,
        max_stop_counter: int = 15,
        verbose: bool = True,
        callback: Callable = None,
        rng: np.random.Generator = None, 
    ):
        """
        Set `max_iterations` to zero to skip optimization
        See https://arxiv.org/abs/1805.05188 theorem 3 and 5
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
            starting_values = np.clip(starting_values, min_value, np.inf)
            starting_values = np.sqrt(starting_values) / scale
        
        # Average-information REML
        state = np.power(starting_values, 2)
        running_mean = state.copy()
        numerator = np.zeros(state.size)
        denominator = np.zeros(state.size)
        stop_counter = 0
        if callback is not None: callback(running_mean * np.power(scale, 2))
        for itt in range(max_iterations):
            gradient, _, _, average_information = tslmm._reml_stochastic_average_information( # change to gradient
                *state, y, X, covariance, preconditioner, 
                trace_samples=trace_samples, indices=indices, rng=rng,
            )
            state += np.linalg.solve(average_information, gradient)
            state = np.clip(state, min_value, np.inf)  # force positive
            rel_change = np.abs((state - running_mean) / running_mean)
            if np.all(rel_change < stop_rtol): stop_counter += 1
            running_mean = state.copy()
            if verbose: print(f"Iteration {itt}: {(running_mean * np.power(scale, 2)).round(2)}, {np.linalg.norm(gradient)}")
            if callback is not None: callback(running_mean * np.power(scale, 2))
            if stop_counter > max_stop_counter: break 
            # TODO: stopping condition based on change in running mean
            # TODO: fit a quadratic using gradient from last K iterations to get better estimate

        _, fixed_effects, residuals, _ = tslmm._reml_stochastic_average_information( # change to gradient for rollback
            *running_mean, y, X, covariance, preconditioner, 
            trace_samples=trace_samples, indices=indices, rng=rng
        )
        fixed_effects = np.linalg.solve(R, fixed_effects) * scale
        running_mean *= np.power(scale, 2)
        residuals *= scale
        residuals += mean

        return running_mean, fixed_effects, residuals

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
            return covariance(0, 1, test_vectors, rows=indices, cols=indices, centre=True)

        def _PG(test_vectors):
            return _projection(_G(test_vectors))

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
        preconditioner_rank: int = 20,
        preconditioner_depth: int = 5,
        rng: np.random.Generator = None,
    ):
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
                depth=preconditioner_depth,
                num_vectors= 2 * preconditioner_rank, 
                indices=self.phenotyped_individuals,
                rng=rng,
            )
        self.rng = rng 

    def set_variance_components(
        self,
        variance_components: np.ndarray
        ):
        """
        Set variance component parameters of ARG-LMM to a given value

        :param np.ndarray variance_components: `sigma^2` and `tau^2` (in this order)
        """

        self.fit_variance_components(variance_components, max_iterations=0)

    def fit_variance_components(
        self, 
        variance_components_init: np.ndarray = None,
        method: str = 'adadelta', 
        haseman_elston: bool = False, 
        haseman_elston_samples: int = 200,
        max_iterations: int = 50,
        max_stop_counter: int = 15,
        sgd_samples: int = 100,
        sgd_decay: float = 0.1, # 
        sgd_epsilon: float = 1e-4,  # if this is too large SGD will diverge
        verbose: bool = True,
        ):
        """
        Fit variance component parameters of ARG-LMM

        :param np.ndarray variance_components_init: Initial estimates of `sigma^2` and `tau^2` (in this order).
        :param method: Either 'adadelta' or 'ai' (average information).
        :param bool haseman_elston: Use Haseman-Elston Initialization.
        :param int haseman_elston_samples: Number of test vectors for Haseman-Elston regression.
        :param int max_iterations: Max iterations of the optimization.
        :param int max_stop_counter: Stop optimization when stop counter hits this number.
        :param int sgd_samples: Number of test vectors for gradient estimation.
        :param float sgd_decay: Decay parameter of AdaDelta.
        :param float sgd_epsilon: Epsilon parameter of AdaDelta.
        :param bool verbose: Print intermediate parameters.
        """

        if haseman_elston:
            variance_components_init = self._haseman_elston_regression(
                phenotypes=self.phenotypes,
                covariates=self.covariates,
                covariance=self.covariance,
                indices=self.phenotyped_individuals,
                trace_samples=haseman_elston_samples,
                rng=self.rng,
            )

        self._optimization_trajectory = []
        callback_fn = lambda x: self._optimization_trajectory.append(x)
        if method == 'adadelta':
            self.variance_components, self.fixed_effects, self.residuals = \
                self._reml_stochastic_optimize(
                    starting_values=variance_components_init,
                    phenotypes=self.phenotypes,
                    covariates=self.covariates,
                    covariance=self.covariance,
                    preconditioner=self.preconditioner,
                    indices=self.phenotyped_individuals,
                    max_iterations=max_iterations,
                    trace_samples=sgd_samples,
                    epsilon=sgd_epsilon,
                    decay=sgd_decay,
                    verbose=verbose,
                    rng=self.rng,
                    callback=callback_fn,
                )

        if method == "ai":
            self.variance_components, self.fixed_effects, self.residuals = \
                self._reml_stochastic_optimize_ai(
                    starting_values=variance_components_init,
                    phenotypes=self.phenotypes,
                    covariates=self.covariates,
                    covariance=self.covariance,
                    preconditioner=self.preconditioner,
                    indices=self.phenotyped_individuals,
                    max_iterations=max_iterations, 
                    trace_samples=sgd_samples,
                    verbose=verbose,
                    rng=self.rng,
                    callback=callback_fn,
                )
            # average over past iterations
            avg_variance_components = np.vstack(
                    self._optimization_trajectory[-max_stop_counter:]
                    ).mean(axis=0)
            self._optimization_trajectory.append(avg_variance_components)
            if verbose: print(f"Final variance component values: {avg_variance_components.round(4)}")
            self.variance_components, self.fixed_effects, self.residuals = \
                self._reml_stochastic_optimize_ai(
                    starting_values=avg_variance_components,
                    phenotypes=self.phenotypes,
                    covariates=self.covariates,
                    covariance=self.covariance,
                    preconditioner=self.preconditioner,
                    indices=self.phenotyped_individuals,
                    max_iterations=0, 
                    trace_samples=sgd_samples,
                    verbose=verbose,
                    rng=self.rng,
                    callback=callback_fn,
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
    
## TODO: borrow some ideas from https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009659

