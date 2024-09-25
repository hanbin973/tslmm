"""
Check that trace estimator improves with increasing numbers of matvecs
"""

import os
import numba
import numpy as np
import msprime
import scipy

from tslmm.trace_estimators import xtrace
from tslmm.tslmm import CovarianceModel, LowRankPreconditioner
from tslmm.tslmm import _explicit_covariance_matrix

import matplotlib.pyplot as plt

if __name__ == "__main__":

    fig_dir = os.path.join(os.path.dirname(__file__), "figs")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    num_threads = 5
    numba.set_num_threads(num_threads)
    rng = np.random.default_rng(seed=1)
    
    n_samples = 1000
    n_pops = 10
    s_length = 1e5
    ne = np.linspace(1e3, 1e5, n_pops)
    island_model = msprime.Demography.island_model(ne, 1e-5 / n_pops)
    for i in range(1, n_pops):
        island_model.add_mass_migration(time=2*np.max(ne), source=i, dest=0, proportion=1.0)
    ts = msprime.sim_ancestry(
        samples={f"pop_{i}": n_samples // n_pops for i in range(n_pops)},
        recombination_rate=1e-8,
        sequence_length=s_length,
        demography=island_model,
        random_seed=1024,
    )

    sigma = 1.5
    tau = 0.5
    mu = 1e-10
    precision = np.linalg.inv(_explicit_covariance_matrix(sigma, tau, ts, mu))
    actual_trace = np.sum(precision.diagonal())

    covr = CovarianceModel(ts, mu)
    prec = LowRankPreconditioner(covr, rank=100, num_vectors=200)
    M = lambda x: prec(sigma, tau, x)
    matvec = lambda x: covr.solve(sigma, tau, x, preconditioner=M)
    
    samples_grid = np.arange(2, 100)
    stream = [xtrace(matvec, covr.dim, samples, rng=rng) for samples in samples_grid]
    est = np.array([s for s, e in stream])
    err = np.array([e for s, e in stream])
    
    plt.scatter(samples_grid, est, s=4, c="red")
    plt.vlines(samples_grid, est - 1.96 * err, est + 1.96 * err, color="red")
    plt.axhline(y=actual_trace, linestyle="--", color="black")
    plt.title("Xtrace convergence")
    plt.xlabel("Number of random vectors")
    plt.ylabel("Trace estimate")
    plt.savefig("figs/check_trace_estimators.png")
