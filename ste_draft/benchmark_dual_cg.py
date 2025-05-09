"""
Precision matrix for edge effects and precision for individual traits is related by
1 / s * (Z (s / t * L L')^{-1} Z' + I)^{-1} = 1 / s * (I - Z (s/t * L L' + Z' Z)^{-1} Z')
by Sherman-Morrison-Woodbury
"""

import tskit
import time
import msprime
import numpy as np
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
    

def run_simulation(num_samples, sequence_length, rank, seed=None):
    """
    Solve system with CG, record work
    """
    global _num_iter

    rng = np.random.default_rng(seed)

    ne = 1e4
    tau2 = 0.5
    sigma2 = 1.5
    ts = msprime.sim_ancestry(
        samples=num_samples,
        recombination_rate=1e-8,
        sequence_length=sequence_length,
        population_size=ne,
        random_seed=seed,
        #model=msprime.SmcKApproxCoalescent(),
    )
    num_edges = ts.num_edges
    covariance = TraitCovariance(ts, mutation_rate=1e-8)
    #preconditioner = NystromPreconditioner(covariance, rank=rank, samples=rank * 2, seed=seed + 1)
    _, y_bar, y = covariance.simulate(sigma2, tau2, rng)
    dim = (covariance.dim, covariance.dim)
    A = sigma2 / tau2 * covariance.L @ covariance.L.T + covariance.Z.T @ covariance.Z
    Zy = covariance.Z.T @ y

    # test: check dual form above
    mat = covariance.as_matrix(sigma2, tau2)
    test1 = np.linalg.solve(mat, y)
    test2 = (y - covariance.Z @ sparse.linalg.spsolve(A, Zy)) / sigma2
    np.testing.assert_allclose(test1, test2)

    # test: CG should converge in 1 iteration with exact inverse
    Mlu = sparse.linalg.splu(A)
    M = sparse.linalg.LinearOperator(A.shape, lambda y: Mlu.solve(y))
    solution, iterations, run_time, resid_norm = run_cg(A, M, Zy)
    assert iterations == 1

    # actual application
    st = time.time()
    Milu = sparse.linalg.spilu(A, drop_tol=1e-1)
    M = sparse.linalg.LinearOperator(A.shape, lambda y: Milu.solve(y))
    en = time.time()
    precond_time = en - st

    solution, iterations, run_time, resid_norm = run_cg(A, M, Zy)
    print(
        f"(Rank {rank} preconditioner; {num_samples} diploids; {num_edges} edges; {sequence_length} bp)\n" 
        f"Preconditioner: {run_time:.2f} seconds\n"
        f"CG iterations: {iterations}; runtime: {run_time:.2f} seconds; residual norm: {resid_norm}"
    )

    solution, iterations, run_time, resid_norm = run_cg(A, None, Zy)
    print(
        f"(No preconditioner; {num_samples} diploids; {num_edges} edges; {sequence_length} bp) " 
        f"CG iterations: {iterations}; runtime: {run_time:.2f} seconds; residual norm: {resid_norm}"
    )


# ------------ #

if __name__ == "__main__":
    run_simulation(100, 1e5, rank=100, seed=1)
