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
import time
import numba
import msprime
import numpy as np
import ray
import scipy.sparse as sparse
import scipy
import sys
import os

tsblup_path = os.path.join(os.path.dirname(__file__), "../../tsblup") # assuming we're in subdir of tsblup repo
sys.path.append(tsblup_path)
import tsblup.operations as operations
import tsblup.matrices as matrices

# --- linear operators --- #

_i1r = numba.types.Array(numba.types.int32, 1, 'C', readonly=True)
_f1r = numba.types.Array(numba.types.float64, 1, 'C', readonly=True)
_f2w = numba.types.Array(numba.types.float64, 2, 'C', readonly=False)
_void = numba.types.void


class TraitCovariance:
    """
    Covariance contribution from tree sequence: given `y = Z u + e` where `u ~
    N(0, tau L^{-1} L^{-T})` and `e ~ N(0, sigma I)`, then `y ~ N(0, tau Z
    L^{-1} L^{-T} Z^T + sigma I)`
    """

    @staticmethod
    @numba.njit(_void(_i1r, _i1r, _f1r, _f2w))
    def backward_solve(Lp, Li, Lx, y):
        """
        `L` is lower-triangular Cholesky factor in CSC format: solve `L x = y`.
        `y` is updated in-place.
        """
        r, c = y.shape
        x = y
        for i in numba.prange(c):
            for j in range(0, r):
                x[j, i] /= Lx[Lp[j]]
                for p in range(Lp[j] + 1, Lp[j + 1]):
                    x[Li[p], i] -= Lx[p] * x[j, i]

    @staticmethod
    @numba.njit(_void(_i1r, _i1r, _f1r, _f2w))
    def forward_solve(Lp, Li, Lx, y):
        """
        `L` is lower-triangular Cholesky factor in CSC format: solve `L' x = y`.
        `y` is updated in-place.
        """
        r, c = y.shape
        x = y
        for i in numba.prange(c):
            for j in range(r - 1, -1, -1):
                for p in range(Lp[j] + 1, Lp[j + 1]):
                    x[j, i] -= Lx[p] * x[Li[p], i]
                x[j, i] /= Lx[Lp[j]]

    def __init__(self, tree_sequence):
        ts = operations.split_upwards(tree_sequence)
        self.dim = ts.num_individuals
        self.Z = matrices.edge_individual_matrix(ts).T
        self.L = scipy.sparse.identity(ts.num_edges) - matrices.edge_adjacency(ts).T
        self.L = self.L.T  # csc

    def __call__(self, sigma, tau, y):
        """
        Matrix-vector product
        """
        Zy = self.Z.T @ y
        self.backward_solve(self.L.indptr, self.L.indices, self.L.data, Zy if y.ndim > 1 else Zy[:, None])
        self.forward_solve(self.L.indptr, self.L.indices, self.L.data, Zy if y.ndim > 1 else Zy[:, None])
        Zy = self.Z @ Zy * tau + y * sigma
        return Zy

    def as_matrix(self, sigma, tau):
        """
        For testing only
        """
        Zd = np.asarray(self.Z.todense())
        Ld = np.asarray(self.L.todense())
        Sigma = tau * Zd @ np.linalg.solve(Ld @ Ld.T, Zd.T) + np.eye(Zd.shape[0]) * sigma
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
    # for the trace estimation

    def __init__(self, A, M, maxiter=None):  # theoretically these are shared among workers
        self.A = A
        self.M = M
        self.maxiter = maxiter

    def solve(self, x):
        assert x.ndim == 1, f"Wrong shape {x.shape}"
        y, _ = sparse.linalg.cg(self.A, x, M=self.M, maxiter=self.maxiter)
        return y


# --- exact and stochastic score function --- #

def exact_loglikelihood(sigma, tau, y, Z, L):
    Zd = np.asarray(covariance.Z.todense())
    Ld = np.asarray(covariance.L.todense())
    Sigma = tau * Zd @ np.linalg.solve(Ld @ Ld.T, Zd.T) + np.eye(Zd.shape[0]) * sigma
    sign, logdet = np.linalg.slogdet(Sigma)
    assert sign > 0
    deviance = logdet + np.dot(y, np.linalg.solve(Sigma, y))
    return -deviance / 2

def exact_gradient(sigma, tau, y, Z, L):
    Zd = np.asarray(covariance.Z.todense())
    Ld = np.asarray(covariance.L.todense())
    S = Zd @ np.linalg.solve(Ld @ Ld.T, Zd.T)
    Sigma = tau * S + np.eye(Zd.shape[0]) * sigma
    r = np.linalg.solve(Sigma, y)
    grad_sigma = np.sum(np.linalg.inv(Sigma).diagonal()) - np.sum(r ** 2)
    grad_tau = np.sum(np.linalg.solve(Sigma, S).diagonal()) - np.dot(r, S @ r)
    return -grad_sigma / 2, -grad_tau / 2

def stochastic_gradient(sigma, tau, y, covariance, preconditioner, samples=1, cg_maxiter=None, rng=None, variance_reduction=True):
    """
    Use the (preconditioned) stochastic trace estimator from Section 3.3 in:
    https://proceedings.mlr.press/v162/wenger22a/wenger22a.pdf
    """

    start = time.time()
    if rng is None: rng = np.random.default_rng()

    # set up linear system
    dim = (covariance.dim, covariance.dim)
    A = sparse.linalg.LinearOperator(dim, lambda y: covariance(sigma, tau, y))
    M = sparse.linalg.LinearOperator(dim, lambda y: preconditioner(sigma, tau, y))
    solution, info = sparse.linalg.cg(A, y, M=M)
    assert info == 0, "CG failed"
    solver = DistributedConjugateGradient.remote(A, M, maxiter=cg_maxiter)

    # approx grad wrt sigma
    test_vectors = rng.normal(size=(covariance.dim, samples))
    test_vectors /= np.sqrt(np.sum(test_vectors ** 2, 0))[None, :]  # unit circle
    sketch = np.column_stack(ray.get([solver.solve.remote(x) for x in test_vectors.T]))
    sigma_grad = np.sum(sketch * test_vectors) * preconditioner.dim / samples - np.sum(solution ** 2)
    if variance_reduction:
        sigma_control = preconditioner.grad_sigma(sigma, tau, test_vectors)
        sigma_grad += preconditioner.grad_sigma(sigma, tau) - preconditioner.dim * sigma_control

    # approx grad wrt tau
    test_vectors = rng.normal(size=(covariance.dim, samples))
    test_vectors /= np.sqrt(np.sum(test_vectors ** 2, 0))[None, :]  # unit circle
    sketch = covariance(0, 1, test_vectors)  # parallelized
    sketch = np.column_stack(ray.get([solver.solve.remote(x) for x in sketch.T]))
    rotation = covariance(0, 1, solution)
    tau_grad = np.sum(sketch * test_vectors) * preconditioner.dim / samples - np.sum(solution * rotation)
    if variance_reduction:
        tau_control = preconditioner.grad_tau(sigma, tau, test_vectors)
        tau_grad += preconditioner.grad_tau(sigma, tau) - preconditioner.dim * tau_control

    end = time.time()
    print(f"Likelihood: {end - start:.2f} seconds")

    return np.array([-sigma_grad / 2, -tau_grad / 2])


# --- verification stuff --- #

def _verify_preconditioner(tau, sigma, preconditioner, seed):
    import numdifftools as nd
    np.random.seed(seed)
    y = np.random.randn(covariance.dim)
    explicit_preconditioner = preconditioner.as_matrix(sigma, tau)

    # check solve(M, y)
    ck_product = np.linalg.solve(explicit_preconditioner, y)
    np.testing.assert_allclose(ck_product, preconditioner(sigma, tau, y))

    # check logdet(M)
    ck_sign, ck_logdet = np.linalg.slogdet(explicit_preconditioner)
    assert ck_sign > 0
    np.testing.assert_allclose(ck_logdet, preconditioner.logdet(sigma, tau))

    # check trace(solve(M, dM/ds))
    jac_sigma = nd.Jacobian(lambda sigma: preconditioner.as_matrix(sigma, tau))(sigma)
    jac_sigma = jac_sigma.reshape(-1, preconditioner.dim)
    ck_grad_sigma = np.linalg.solve(explicit_preconditioner, jac_sigma)
    ck_grad_sigma = np.sum(ck_grad_sigma.diagonal())
    np.testing.assert_allclose(ck_grad_sigma, preconditioner.grad_sigma(sigma, tau))

    # check trace(solve(M, dM/dt))
    jac_tau = nd.Jacobian(lambda tau: preconditioner.as_matrix(sigma, tau))(tau)
    jac_tau = jac_tau.reshape(-1, preconditioner.dim)
    ck_grad_tau = np.linalg.solve(explicit_preconditioner, jac_tau)
    ck_grad_tau = np.sum(ck_grad_tau.diagonal())
    np.testing.assert_allclose(ck_grad_tau, preconditioner.grad_tau(sigma, tau))


def _verify_covariance(tau, sigma, covariance, seed):
    np.random.seed(seed)
    y = np.random.randn(covariance.dim)
    explicit_covariance = covariance.as_matrix(sigma, tau)

    # check Sigma @ y
    ck_product = explicit_covariance @ y
    np.testing.assert_allclose(ck_product, covariance(sigma, tau, y))


def _verify_exact_score(tau, sigma, covariance, seed):
    import numdifftools as nd
    np.random.seed(seed)
    y = np.random.randn(covariance.dim)
    sigma_grad = nd.Derivative(lambda sigma: exact_loglikelihood(sigma, tau, y, covariance.Z, covariance.L), n=1, step=1e-4)
    tau_grad = nd.Derivative(lambda tau: exact_loglikelihood(sigma, tau, y, covariance.Z, covariance.L), n=1, step=1e-4)
    ck_score = np.array([sigma_grad(sigma), tau_grad(tau)])
    np.testing.assert_allclose(ck_score, exact_gradient(sigma, tau, y, covariance.Z, covariance.L))
    stoch_score = stochastic_gradient(sigma, tau, y, covariance, preconditioner, samples=100)
    print(ck_score, stoch_score)



# --- test against exact on a small example --- #

num_threads = 4
numba.set_num_threads(num_threads)
ray.init(num_cpus=num_threads)

ts = msprime.sim_ancestry(
    samples=100,
    recombination_rate=1e-8,
    sequence_length=1e5,
    population_size=1e4,
    random_seed=1024,
)
covariance = TraitCovariance(ts)
preconditioner = NystromPreconditioner(covariance, rank=10, samples=20, seed=1)

# annoyingly, this is needed for ray to work-- we can't serialize a non-module import
del sys.modules["tsblup"] 
del operations
del matrices

# don't do this check with >100 samples
_verify_preconditioner(0.9, 0.25, preconditioner, 1024)
_verify_covariance(0.9, 0.25, covariance, 1024)
_verify_exact_score(0.9, 0.25, covariance, 1024)
assert False

# does it work?
def compare_exact_vs_stochastic(tau, sigma, y, covariance, preconditioner, rng=None, samples=1):
    #exact = exact_gradient(sigma, tau, y, covariance.Z, covariance.L)
    exact = np.array([np.nan, np.nan])
    stochastic = stochastic_gradient(sigma, tau, y, covariance, preconditioner, samples=samples, rng=rng, variance_reduction=False)
    stochastic_vr = stochastic_gradient(sigma, tau, y, covariance, preconditioner, samples=samples, rng=rng, variance_reduction=True)
    return exact, stochastic, stochastic_vr

rng = np.random.default_rng(1024)
y = rng.normal(size=covariance.dim)
sigma_grid = np.linspace(0.1, 2.0, 25)
tau_grid = np.linspace(0.1, 2.0, 25)

sigma_exact = []
sigma_approx = []
sigma_approx_vr = []
for sigma in sigma_grid:
    exact, stochastic, stochastic_vr = compare_exact_vs_stochastic(sigma, np.median(tau_grid), y, covariance, preconditioner, rng=rng)
    sigma_exact.append(exact)
    sigma_approx.append(stochastic)
    sigma_approx_vr.append(stochastic_vr)
sigma_exact = np.array(sigma_exact).reshape(-1, 2)
sigma_approx = np.array(sigma_approx).reshape(-1, 2)
sigma_approx_vr = np.array(sigma_approx_vr).reshape(-1, 2)

tau_exact = []
tau_approx = []
tau_approx_vr = []
for tau in tau_grid:
    exact, stochastic, stochastic_vr = compare_exact_vs_stochastic(np.median(sigma_grid), tau, y, covariance, preconditioner, rng=rng)
    tau_exact.append(exact)
    tau_approx.append(stochastic)
    tau_approx_vr.append(stochastic_vr)
tau_exact = np.array(tau_exact).reshape(-1, 2)
tau_approx = np.array(tau_approx).reshape(-1, 2)
tau_approx_vr = np.array(tau_approx_vr).reshape(-1, 2)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].plot(sigma_grid, sigma_exact[:, 0], label="exact")
axs[0].scatter(sigma_grid, sigma_approx[:, 0], label="approx")
axs[0].scatter(sigma_grid, sigma_approx_vr[:, 0], label="approx-vr")
axs[0].set_ylabel("gradient wrt sigma")
axs[0].set_xlabel("sigma")
axs[0].legend()
axs[1].plot(tau_grid, tau_exact[:, 0], label="exact")
axs[1].scatter(tau_grid, tau_approx[:, 0], label="approx")
axs[1].scatter(tau_grid, tau_approx_vr[:, 0], label="approx-vr")
axs[1].set_ylabel("gradient wrt tau")
axs[1].set_xlabel("tau")
axs[1].legend()
fig.tight_layout()
plt.savefig("ste_check.png")

