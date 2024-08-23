
import tskit
import time
import msprime
import numpy as np
import scipy.sparse as sparse

from linear_operators import TraitCovariance, NystromPreconditioner


def run_cg(sigma, tau, covariance, preconditioner, y):
    """
    Solve `A x = y` with conjugate gradient and preconditioner `M`
    """
    start = time.time()
    M = None if preconditioner is None else lambda x: preconditioner(sigma, tau, x)
    solution, iterations, converged = covariance.solve(sigma, tau, y, preconditioner=M)
    assert converged
    end = time.time()
    run_time = float(end - start)
    residual_norm = np.linalg.norm(covariance(sigma, tau, solution) - y, 2)
    return solution, iterations, run_time, residual_norm
    

def run_simulation(num_samples, sequence_length, rank, seed=None):
    """
    Solve system with CG, record work
    """
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
    covariance = TraitCovariance(ts, mutation_rate=1e-10)
    preconditioner = NystromPreconditioner(covariance, rank=rank, samples=rank * 2, seed=seed + 1)
    _, y_bar, y = covariance.simulate(sigma2, tau2, rng)

    solution, iterations, run_time, resid_norm = run_cg(sigma2, tau2, covariance, preconditioner, y)
    print(
        f"(Rank {rank} preconditioner; {num_samples} diploids; {sequence_length} bp) " 
        f"CG iterations: {iterations}; runtime: {run_time:.2f} seconds; residual norm: {resid_norm}"
    )

    solution, iterations, run_time, resid_norm = run_cg(sigma2, tau2, covariance, None, y)
    print(
        f"(No preconditioner; {num_samples} diploids; {sequence_length} bp) " 
        f"CG iterations: {iterations}; runtime: {run_time:.2f} seconds; residual norm: {resid_norm}"
    )


# ------------ #

if __name__ == "__main__":
    run_simulation(1000, 1e6, rank=100, seed=1)
