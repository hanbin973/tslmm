"""
CG scaling
"""

import tskit
import time
import numba
import msprime
import numpy as np
import ray
import scipy.sparse as sparse
import scipy
import sys
import os

tsblup_path = os.path.join(os.path.dirname(__file__), "../tsblup")
sys.path.append(tsblup_path)
import tsblup.operations as operations
import tsblup.matrices as matrices

numba.set_num_threads(20)
#ray.init(num_cpus=8)

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
    @numba.njit(_f2w(_i1r, _i1r, _f1r, _f2r))
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
        return x

    @staticmethod
    @numba.njit(_f2w(_i1r, _i1r, _f1r, _f2r))
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
        return x

    def __init__(self, tree_sequence):
        ts = operations.split_upwards(tree_sequence)
        edge_area = (ts.edges_right - ts.edges_left) * \
            (ts.nodes_time[ts.edges_parent] - ts.nodes_time[ts.edges_child])
        self.dim = ts.num_individuals
        self.Z = matrices.edge_individual_matrix(ts).T
        self.L = scipy.sparse.identity(ts.num_edges) - matrices.edge_adjacency(ts).T
        self.L = self.L.T @ scipy.sparse.diags_array(1 / np.sqrt(edge_area))
        self.L.sort_indices()

    def __call__(self, sigma, tau, y):
        """
        Matrix-vector product
        """
        assert self.L.has_sorted_indices
        if y.ndim == 1: y = y.reshape(-1, 1)
        Zy = self.Z.T @ y
        Zy = self.backward_solve(self.L.indptr, self.L.indices, self.L.data, Zy)
        Zy = self.forward_solve(self.L.indptr, self.L.indices, self.L.data, Zy)
        Zy = self.Z @ Zy * tau + y * sigma
        return Zy

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
        #ck_Sigma = tau * Zd @ sparse.linalg.spsolve(self.L @ self.L.T, Zd.T) + np.eye(self.dim) * sigma
        #np.testing.assert_allclose(Sigma, ck_Sigma)
        return Sigma


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
        start = time.time()
        self.D, self.U = self._rand_eigh(covariance, rank, samples, seed)
        end = time.time()
        print(f"Preconditioner: {end - start:.2f} seconds")

    def __call__(self, sigma, tau, y):
        """
        Inverse-vector product: section 17.2 in https://arxiv.org/pdf/2002.01387
        """
        S = (self.D * tau + sigma) / (np.min(self.D) * tau + sigma)
        My = (self.U.T @ y) * (1 - 1 / S)
        return y - self.U @ My

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
        trace(solve(preconditioner, deriv(preconditioner, sigma)))
        """
        spectrum = self.D * tau + sigma
        norm = np.min(spectrum)
        spectrum /= norm
        grad_sigma = (1 - spectrum) / spectrum / norm
        if y is None:  # exact
            return np.sum(grad_sigma)
        else:  # Hutchinson estimator for control variate
            assert y.ndim == 2
            control = (self.U.T @ y) * grad_sigma[:, np.newaxis]
            control = self.U @ control
            return np.sum(y * control) / y.shape[1]

    def grad_tau(self, sigma, tau, y=None):
        """
        trace(solve(preconditioner, deriv(preconditioner, tau)))
        """
        spectrum = self.D * tau + sigma
        norm = np.min(spectrum)
        spectrum /= norm
        grad_tau = (self.D - np.min(self.D) * spectrum) / spectrum / norm
        if y is None:  # exact
            return np.sum(grad_tau)
        else:  # Hutchinson estimator for control variate
            assert y.ndim == 2
            control = (self.U.T @ y) * grad_tau[:, np.newaxis]
            control = self.U @ control
            return np.sum(y * control) / y.shape[1]


@ray.remote
class DistributedConjugateGradient:
    # TODO: in Wenger et al, they use a fixed number of CG iterations
    # for the trace estimation, but it's not clear how this would lead to unbiased estimates

    def __init__(self, A, M, maxiter=None):  # theoretically these are shared among workers
        self.A = A
        self.M = M
        self.maxiter = maxiter

    def solve(self, x):
        assert x.ndim == 1, f"Wrong shape {x.shape}"
        y, _ = sparse.linalg.cg(self.A, x, M=self.M, maxiter=self.maxiter)
        return y


## ----

def sim_traits(ts, edge_sd, obs_sd, rng):
    edge_area = (ts.edges_right - ts.edges_left) * (ts.nodes_time[ts.edges_parent] - ts.nodes_time[ts.edges_child])
    u = rng.normal(size=ts.num_edges) * edge_sd * np.sqrt(edge_area)
    Z = matrices.edge_individual_matrix(ts).T
    E = matrices.edge_adjacency(ts).T
    I = scipy.sparse.eye(E.shape[0])
    U = I - E
    L = U.T
    L.sort_indices()
    r = TraitCovariance.forward_solve(L.indptr, L.indices, L.data, u[:, None]).flatten()
    e = rng.normal(size=Z.shape[0]) * obs_sd
    #r2 = scipy.sparse.linalg.spsolve(U, u)
    #np.testing.assert_allclose(r2, r)
    out = Z.dot(r) + e
    return out


def run_cg(A, M, y):

    global _num_iter
    _num_iter = 0
    def _count_iter(xk):
        global _num_iter
        _num_iter += 1

    start = time.time()
    solution, info = sparse.linalg.cg(A, y, callback=_count_iter, M=M)
    assert info == 0, "CG failed"
    residual_norm = np.linalg.norm(A @ solution - y, 2)
    end = time.time()
    iterations = _num_iter
    run_time = float(end - start)
    return solution, iterations, run_time, residual_norm
    

def run_sim(num_samples, sequence_length, rank, seed=None):
    global _num_iter

    rng = np.random.default_rng(seed)

    start = time.time()
    ne = 1e4
    tau2 = 1.0 / ne * 1.0 / sequence_length
    sigma2 = 1.0
    ts = msprime.sim_ancestry(
        samples=num_samples,
        recombination_rate=1e-8,
        sequence_length=sequence_length,
        population_size=ne,
        random_seed=seed,
        #model=msprime.SmcKApproxCoalescent(),
    )
    covariance = TraitCovariance(ts)
    preconditioner = NystromPreconditioner(covariance, rank=rank, samples=rank*2, seed=seed + 1)

    ###DEBUG
    #cov_mat = covariance.as_matrix(sigma2, tau2)
    #import matplotlib.pyplot as plt
    #img = plt.matshow(cov_mat)
    #plt.colorbar(img)
    #plt.savefig("cov_matrix_dense.png")
    #plt.clf()
    #plt.scatter(np.arange(cov_mat.shape[0]), np.linalg.eigh(cov_mat)[0])
    #plt.savefig("cov_matrix_spec.png")
    #plt.clf()

    y_bar = sim_traits(operations.split_upwards(ts), np.sqrt(tau2), 0.0, rng)
    y = y_bar + rng.normal(0, np.sqrt(sigma2), size=y_bar.size)
    end = time.time()
    print("trait RMSE", np.sqrt(np.mean(y_bar**2)), np.sqrt(np.mean((y-y_bar)**2)), np.sqrt(np.mean(y**2)))
    print("trait Var", np.mean(y_bar**2), np.mean(y**2), np.mean((y-y_bar)**2))
    print(f"Simulation time {end - start:.2f}")

    ###DEBUG
    #chol = np.linalg.cholesky(cov_mat) 
    #Y = chol @ rng.normal(size=(chol.shape[0], 1000))
    #varm = []
    #tss = operations.split_upwards(ts)
    #for _ in range(1000):
    #    xx = sim_traits(tss, np.sqrt(tau2), np.sqrt(sigma2), rng)
    #    varm += [xx[0]]
    #print("**", np.std(Y[0, :]))
    #print("***", np.std(varm))
    #assert False
    ###

    dim = (covariance.dim, covariance.dim)
    A = sparse.linalg.LinearOperator(dim, lambda y: covariance(sigma2, tau2, y))
    M = sparse.linalg.LinearOperator(dim, lambda y: preconditioner(sigma2, tau2, y))

    ###DEBUG
    #np.testing.assert_allclose(cov_mat @ y, A @ y)

    solution, iterations, run_time, resid_norm = run_cg(A, M, y)
    print(f"CG iterations: {iterations}; runtime: {run_time:.2f} seconds; residual norm: {resid_norm}")

    solution, iterations, run_time, resid_norm = run_cg(A, None, y)
    print(f"CG iterations: {iterations}; runtime: {run_time:.2f} seconds; residual norm: {resid_norm}")

#run_sim(100000, 1e7, rank=10, seed=1)
run_sim(100000, 5e7, rank=10, seed=1)
