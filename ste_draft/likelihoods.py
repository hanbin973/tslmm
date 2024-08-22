"""
See 'linear_operators.py' for overview
"""

import time
import numpy as np
import scipy.sparse as sparse
import ray

from trace_estimators import xtrace, xdiag

#def conjugate_gradient(A : sparse.linalg.LinearOperator, b :




def exact_loglikelihood(sigma, tau, y, covariance):
    Sigma = covariance.as_matrix(sigma, tau)
    sign, logdet = np.linalg.slogdet(Sigma)
    assert sign > 0
    deviance = logdet + np.dot(y, np.linalg.solve(Sigma, y))
    return -deviance / 2


def exact_gradient(sigma, tau, y, covariance):
    S = covariance.as_matrix(0, 1)
    Sigma = covariance.as_matrix(sigma, tau)
    r = np.linalg.solve(Sigma, y)
    grad_sigma = np.sum(np.linalg.inv(Sigma).diagonal()) - np.sum(r ** 2)
    grad_tau = np.sum(np.linalg.solve(Sigma, S).diagonal()) - np.dot(r, S @ r)
    return -grad_sigma / 2, -grad_tau / 2


#@ray.remote
class DistributedConjugateGradient:
    # TODO: in Wenger et al, they use a fixed number of CG iterations
    # for the trace estimation, but it's not clear how this would lead to unbiased estimates

    def __init__(self, A, M, maxiter=None):  # theoretically these are shared among workers
        self.A = A
        self.M = M
        self.maxiter = maxiter

    def solve(self, x):
        global _count_iter
        _count_iter = 0
        def _count_iter_incr(xk):
            global _count_iter
            _count_iter += 1

        assert x.ndim == 1, f"Wrong shape {x.shape}"
        st = time.time()
        y, _ = sparse.linalg.cg(self.A, x, callback=_count_iter_incr, M=self.M, maxiter=self.maxiter)
        en = time.time()
        print(f"DEBUG trace {_count_iter} CG iterations, {en - st:.2f} sec", flush=True)
        return y


def stochastic_gradient(sigma, tau, y, covariance, preconditioner, samples=1, cg_maxiter=None, rng=None, trace_estimator=xtrace, variance_reduction=False):
    """
    Use the (preconditioned) stochastic trace estimator from Section 3.3 in:
    https://proceedings.mlr.press/v162/wenger22a/wenger22a.pdf
    """

    global _count_iter
    _count_iter = 0
    def _count_iter_incr(xk):
        global _count_iter
        _count_iter += 1

    if rng is None: rng = np.random.default_rng()

    # set up linear system
    dim = covariance.dim
    A = sparse.linalg.LinearOperator((dim, dim), lambda y: covariance(sigma, tau, y))
    M = sparse.linalg.LinearOperator((dim, dim), lambda y: preconditioner(sigma, tau, y))
    st = time.time()
    solution, info = sparse.linalg.cg(A, y, callback=_count_iter_incr, M=M)
    en = time.time()
    assert info == 0, "CG failed"
    print(f"DEBUG {_count_iter} CG iterations, {en - st:.2f} sec", flush=True)
    #solver = DistributedConjugateGradient.remote(A, M, maxiter=cg_maxiter)  # this has huge overhead
    solver = DistributedConjugateGradient(A, M, maxiter=cg_maxiter)

    def _sigma_grad_trace(test_vectors):
        if test_vectors.ndim == 1: test_vectors = test_vectors.reshape(-1, 1)
        #sketch = np.column_stack(ray.get([solver.solve.remote(x) for x in test_vectors.T]))
        sketch = np.column_stack([solver.solve(x) for x in test_vectors.T])
        if variance_reduction: sketch -= preconditioner.grad_sigma(sigma, tau, test_vectors)
        return sketch

    sigma_grad_trace = sparse.linalg.LinearOperator((dim, dim), lambda y: _sigma_grad_trace(y))
    sigma_grad_trace, _ = trace_estimator(sigma_grad_trace, samples)
    if variance_reduction: sigma_grad_trace += preconditioner.grad_sigma(sigma, tau)
    sigma_grad = sigma_grad_trace - np.sum(solution ** 2)

    def _tau_grad_trace(test_vectors):
        if test_vectors.ndim == 1: test_vectors = test_vectors.reshape(-1, 1)
        sketch = covariance(0, 1, test_vectors)  # d(covariance)/d(tau)
        #sketch = np.column_stack(ray.get([solver.solve.remote(x) for x in sketch.T]))
        sketch = np.column_stack([solver.solve(x) for x in test_vectors.T])
        if variance_reduction: sketch -= preconditioner.grad_tau(sigma, tau, test_vectors)
        return sketch

    tau_grad_trace = sparse.linalg.LinearOperator((dim, dim), lambda y: _tau_grad_trace(y))
    tau_grad_trace, _ = trace_estimator(tau_grad_trace, samples)
    if variance_reduction: tau_grad_trace += preconditioner.grad_tau(sigma, tau)
    tau_grad = tau_grad_trace - np.sum(solution * covariance(0, 1, solution).flatten())

    return np.array([-sigma_grad / 2, -tau_grad / 2])


def genetic_values(sigma, tau, y, covariance, preconditioner, diag_samples=0):
    """
    y = r + e, r ~ N(0, t Z(LL')^{-1}Z'), e ~ N(0, sI)

    Q = Z(LL')^{-1}Z'
    M = Q^{-1} / t + I / s

    E[r|y] = M^{-1} y / s = (Q^{-1} / t + I / s)^{-1} y / s
           = (I - s (Q t + I s)^{-1}) y

    V[r|y] = M^{-1}
           = I s + s ** 2 (t Z(LL')^{-1}Z' + I s)^{-1}
    """

    global _count_iter
    _count_iter = 0
    def _count_iter_incr(xk):
        global _count_iter
        _count_iter += 1

    dim = covariance.dim
    A = sparse.linalg.LinearOperator((dim, dim), lambda y: covariance(sigma, tau, y))
    M = sparse.linalg.LinearOperator((dim, dim), lambda y: preconditioner(sigma, tau, y))
    st = time.time()
    solution, info = sparse.linalg.cg(A, y, callback=_count_iter_incr, M=M)
    en = time.time()
    assert info == 0, "CG failed"
    print(f"DEBUG {_count_iter} CG iterations, {en - st:.2f} sec", flush=True)

    # expected value
    Ey = y - sigma * solution

    # estimate variance with stochastic estimator
    solver = DistributedConjugateGradient(A, M, maxiter=None)
    def _covariance_diag(test_vectors):
        if test_vectors.ndim == 1: test_vectors = test_vectors.reshape(-1, 1)
        sketch = np.column_stack([solver.solve(x) for x in test_vectors.T])
        return sketch

    if diag_samples > 0:
        covariance_diag = sparse.linalg.LinearOperator((dim, dim), lambda y: _covariance_diag(y))
        covariance_diag = xdiag(covariance_diag, diag_samples)
        Vy = sigma - sigma ** 2 * covariance_diag
    else:
        Vy = np.full(dim, np.nan)

    return Ey, Vy

