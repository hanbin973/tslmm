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


def exact_loglikelihood_reml(sigma: float, tau: float, y: np.ndarray, X: np.ndarray, covariance: TraitCovariance, use_qr: bool = True) -> float:
    """
    log |G(sigma, tau)| + log |X' G(sigma, tau)^{-1} X| + (y - X b_hat)' G(sigma, tau)^{-1} (y - X b_hat)
    b_hat = (X' G(sigma, tau)^{-1} X)^{-1} X' G(sigma, tau)^{-1} y
    """
    G = covariance.as_matrix(sigma, tau)
    # L = np.linalg.cholesky(G) << solve via a factorisation
    sign, Gldet = np.linalg.slogdet(G)
    assert sign > 0
    if use_qr:
        X, R = np.linalg.qr(X)
        Gldet += 2 * np.sum(np.log(np.abs(R.diagonal())))
    GinvX = np.linalg.solve(G, X)
    GinvY = np.linalg.solve(G, y)
    XinvX = X.T @ GinvX
    resid = y - X @ np.linalg.solve(XinvX, X.T @ GinvY)
    Ginvr = np.linalg.solve(G, resid)  
    sign, Xldet = np.linalg.slogdet(XinvX)
    assert sign > 0
    deviance = Gldet + Xldet + resid.T @ Ginvr
    return -deviance / 2


def exact_gradient_reml(sigma: float, tau: float, y: np.ndarray, X: np.ndarray, covariance: TraitCovariance, use_qr: bool = True) -> (float, float):
    if use_qr:  # stable, and gradient is invariant to rotation (doesn't depend on R)
        X, _ = np.linalg.qr(X)
    G = covariance.as_matrix(sigma, tau)
    dGdt = covariance.as_matrix(0, 1)
    GinvX = np.linalg.solve(G, X)
    GinvY = np.linalg.solve(G, y)
    XinvX = X.T @ GinvX  # should factorize
    resid = y - X @ np.linalg.solve(XinvX, X.T @ GinvY)
    Ginvr = np.linalg.solve(G, resid)
    dy_ds = X @ np.linalg.solve(XinvX, GinvX.T @ Ginvr)
    dy_dt = X @ np.linalg.solve(XinvX, GinvX.T @ (dGdt @ Ginvr))
    grad_s = np.sum(np.linalg.inv(G).diagonal()) - \
        np.sum(np.linalg.solve(XinvX, GinvX.T @ GinvX).diagonal()) - \
        np.dot(Ginvr.T, Ginvr) + 2 * dy_ds.T @ Ginvr
    grad_t = np.sum(np.linalg.solve(G, dGdt).diagonal()) - \
        np.sum(np.linalg.solve(XinvX, GinvX.T @ (dGdt @ GinvX)).diagonal()) - \
        np.dot(Ginvr.T, dGdt @ Ginvr) + 2 * dy_dt.T @ Ginvr
    return -grad_s / 2, -grad_t / 2


def stochastic_gradient_reml(sigma, tau, y, X, covariance, preconditioner, num_samples=1, rng=None, trace_estimator=xtrace, variance_reduction=False):
    """
    Use the (preconditioned) stochastic trace estimator from Section 3.3 in:
    https://proceedings.mlr.press/v162/wenger22a/wenger22a.pdf

    X can be orthonormalised, e.g. by QR decomposition, which won't change
    the gradient but will be more numerically stable
    """

    if rng is None: rng = np.random.default_rng()

    dim = covariance.dim
    M = lambda y: preconditioner(sigma, tau, y)

    solution, iterations, converged = covariance.solve(sigma, tau, np.hstack([X, np.expand_dims(y, 1)]), preconditioner=M)
    assert converged

    Ginv_Xt, Ginv_y = solution.T[:-1], solution.T[-1]
    Xt_Ginv_X_inv = np.linalg.inv(Ginv_Xt @ X)  # maybe better to factorize
    residuals = y - X @ (Xt_Ginv_X_inv @ (X.T @ Ginv_y))

    solution, iterations, converged = covariance.solve(sigma, tau, residuals, preconditioner=M)
    assert converged

    # all this would be more storage efficient if we had a product fn(X, Y) = X' Ginv Y
    # that didn't require intermediate vectors
    rotation = covariance(0, 1, solution)
    X_Ginv_dGds_r = Ginv_Xt @ solution # solution == covariance(1, 0, solution)
    X_Ginv_dGdt_r = Ginv_Xt @ rotation
    X_Ginv_dGds_X = Ginv_Xt @ Ginv_Xt.T # Ginv_X.T = covariance(1, 0, Ginv_X.T)
    X_Ginv_dGdt_X = Ginv_Xt @ covariance(0, 1, Ginv_Xt.T)
    dy_ds = X @ (Xt_Ginv_X_inv @ X_Ginv_dGds_r)
    dy_dt = X @ (Xt_Ginv_X_inv @ X_Ginv_dGdt_r)
    trc_s = np.sum(Xt_Ginv_X_inv * X_Ginv_dGds_X)
    trc_t = np.sum(Xt_Ginv_X_inv * X_Ginv_dGdt_X)

    def _sigma_grad_trace(test_vectors):
        if test_vectors.ndim == 1: test_vectors = test_vectors.reshape(-1, 1)
        sketch, _, _ = covariance.solve(sigma, tau, test_vectors, preconditioner=M)
        if variance_reduction: sketch -= preconditioner.grad_sigma(sigma, tau, test_vectors)
        return sketch

    sigma_grad_trace, _ = trace_estimator(_sigma_grad_trace, dim, num_samples, rng=rng)
    if variance_reduction: sigma_grad_trace += preconditioner.grad_sigma(sigma, tau)
    sigma_grad = sigma_grad_trace - trc_s - np.sum(solution ** 2) + 2 * np.sum(solution * dy_ds)

    def _tau_grad_trace(test_vectors):
        if test_vectors.ndim == 1: test_vectors = test_vectors.reshape(-1, 1)
        sketch = covariance(0, 1, test_vectors)  # d(covariance)/d(tau)
        sketch, _, _ = covariance.solve(sigma, tau, sketch, preconditioner=M)
        if variance_reduction: sketch -= preconditioner.grad_tau(sigma, tau, test_vectors)
        return sketch

    tau_grad_trace, _ = trace_estimator(_tau_grad_trace, dim, num_samples, rng=rng)
    if variance_reduction: tau_grad_trace += preconditioner.grad_tau(sigma, tau)
    tau_grad = tau_grad_trace - trc_t - np.sum(solution * rotation) + 2 * np.sum(solution * dy_dt)

    return np.array([-sigma_grad / 2, -tau_grad / 2])

