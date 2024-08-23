"""
See 'linear_operators.py' for overview
"""

import time
import numpy as np
import scipy.sparse as sparse

from scipy.sparse.linalg import LinearOperator
from trace_estimators import xtrace, xdiag
from linear_operators import TraitCovariance, NystromPreconditioner


def exact_loglikelihood(sigma: float, tau: float, y: np.ndarray, covariance: TraitCovariance) -> float:
    Sigma = covariance.as_matrix(sigma, tau)
    sign, logdet = np.linalg.slogdet(Sigma)
    assert sign > 0
    deviance = logdet + np.dot(y, np.linalg.solve(Sigma, y))
    return -deviance / 2


def exact_gradient(sigma: float, tau: float, y: np.ndarray, covariance: TraitCovariance) -> (float, float):
    S = covariance.as_matrix(0, 1)
    Sigma = covariance.as_matrix(sigma, tau)
    r = np.linalg.solve(Sigma, y)
    grad_sigma = np.sum(np.linalg.inv(Sigma).diagonal()) - np.sum(r ** 2)
    grad_tau = np.sum(np.linalg.solve(Sigma, S).diagonal()) - np.dot(r, S @ r)
    return -grad_sigma / 2, -grad_tau / 2


def stochastic_gradient(sigma, tau, y, covariance, preconditioner, num_samples=1, rng=None, trace_estimator=xtrace, variance_reduction=False):
    """
    Use the (preconditioned) stochastic trace estimator from Section 3.3 in:
    https://proceedings.mlr.press/v162/wenger22a/wenger22a.pdf
    """

    if rng is None: rng = np.random.default_rng()

    dim = covariance.dim
    M = lambda y: preconditioner(sigma, tau, y)

    solution, iterations, converged = covariance.solve(sigma, tau, y, preconditioner=M)
    assert converged

    def _sigma_grad_trace(test_vectors):
        if test_vectors.ndim == 1: test_vectors = test_vectors.reshape(-1, 1)
        sketch, _, _ = covariance.solve(sigma, tau, test_vectors, preconditioner=M)
        if variance_reduction: sketch -= preconditioner.grad_sigma(sigma, tau, test_vectors)
        return sketch

    sigma_grad_trace, _ = trace_estimator(_sigma_grad_trace, dim, num_samples, rng=rng)
    if variance_reduction: sigma_grad_trace += preconditioner.grad_sigma(sigma, tau)
    sigma_grad = sigma_grad_trace - np.sum(solution ** 2)

    def _tau_grad_trace(test_vectors):
        if test_vectors.ndim == 1: test_vectors = test_vectors.reshape(-1, 1)
        sketch = covariance(0, 1, test_vectors)  # d(covariance)/d(tau)
        sketch, _, _ = covariance.solve(sigma, tau, sketch, preconditioner=M)
        if variance_reduction: sketch -= preconditioner.grad_tau(sigma, tau, test_vectors)
        return sketch

    tau_grad_trace, _ = trace_estimator(_tau_grad_trace, dim, num_samples, rng=rng)
    if variance_reduction: tau_grad_trace += preconditioner.grad_tau(sigma, tau)
    tau_grad = tau_grad_trace - np.sum(solution * covariance(0, 1, solution).flatten())

    return np.array([-sigma_grad / 2, -tau_grad / 2])


def genetic_values(sigma, tau, y, covariance, preconditioner, num_samples=0, rng=None):
    """
    y = g + e, g ~ N(0, tQ), e ~ N(0, sI)

    where 

    Q = Z(LL')^{-1}Z'
    M = Q^{-1} / t + I / s

    then

    E[g|y] = M^{-1} y / s = (Q^{-1} / t + I / s)^{-1} y / s
           = (I - s (Q t + I s)^{-1}) y

    V[g|y] = M^{-1} = (Q^{-1} / t + I / s)^{-1}
           = I s - s ** 2 (t Q + I s)^{-1}
    """

    if rng is None: rng = np.random.default_rng()
    dim = covariance.dim
    M = lambda y: preconditioner(sigma, tau, y)

    solution, iterations, converged = covariance.solve(sigma, tau, y, preconditioner=M)
    assert converged

    # expected value
    Ey = y - sigma * solution
    Vy = np.full(dim, np.nan)

    # estimate variance with stochastic estimator
    def _covariance_diag(test_vectors):
        if test_vectors.ndim == 1: test_vectors = test_vectors.reshape(-1, 1)
        sketch, _, _ = covariance.solve(sigma, tau, test_vectors, preconditioner=M)
        sketch = sigma * (test_vectors - sigma * sketch)
        return sketch

    if num_samples > 0: 
        Vy[:] = xdiag(_covariance_diag, dim, num_samples, rng) 

    return Ey, Vy

