import tskit
import numpy as np

import time
import scipy.sparse as sparse
from linear_operators import TraitCovariance, NystromPreconditioner

_num_iter = 0

def _count_iter(xk):
    global _num_iter
    _num_iter += 1


def run_cg(A, M, y):
    """
    Solve `A x = y` with conjugate gradient and preconditioner `M`
    """
    global _num_iter
    _num_iter = 0

    start = time.time()
    solution, info = sparse.linalg.cg(A, y, callback=_count_iter, M=M)
    assert info == 0, "CG failed"
    residual_norm = np.linalg.norm(A @ solution - y, 2)
    end = time.time()

    iterations = _num_iter
    run_time = float(end - start)
    return solution, iterations, run_time, residual_norm


def predict(y_obs, ind_obs, covariance, sigma2, tau2, rank, rng, seed):
    """
    Solve system with CG, record work
    """
    global _num_iter

    preconditioner = NystromPreconditioner(covariance, rank=rank, samples=rank * 2, seed=seed + 1)
    dim = (covariance.dim, covariance.dim)
    A = sparse.linalg.LinearOperator(dim, lambda y: covariance(sigma2, tau2, y))
    M = sparse.linalg.LinearOperator(dim, lambda y: preconditioner(sigma2, tau2, y))

    # solution, iterations, run_time, resid_norm = run_cg(A, M, y_obs)
    solution, iterations, success = covariance.solve(sigma2, tau2, y_obs, preconditioner=M)
    print(
        f"(Rank {rank} preconditioner; {covariance.dim} diploids; {covariance.kernel_dim} edges) " 
        f"CG iterations: {iterations}; " # runtime: {run_time:.2f} seconds; residual norm: {resid_norm}"
    )

    pred = covariance.cov_all_inds(tau2, solution)
    return pred, iterations, success



