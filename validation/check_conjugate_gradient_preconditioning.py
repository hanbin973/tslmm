"""
Check that preconditioning reduces number of matvecs (CG iterations)
"""

import os
import numba
import numpy as np
import msprime
import scipy

from tslmm.tslmm import tslmm, CovarianceModel, LowRankPreconditioner
from tslmm.tslmm import _explicit_covariance_matrix

import matplotlib.pyplot as plt


def simulate(sigma, tau, tree_sequence, mutation_rate, rng=None, subset=None, center_covariance=False):
    """
    `subset` is only used for centering
    """
    if rng is None: rng = np.random.default_rng()
    if subset is None: subset = np.arange(tree_sequence.num_individuals)
    G = _explicit_covariance_matrix(
        0, tau, tree_sequence, mutation_rate, 
        center_around=subset if center_covariance else None,
    )
    U = np.linalg.cholesky(G)
    X = rng.normal(size=(tree_sequence.num_individuals, 5))
    g = U @ rng.normal(size=tree_sequence.num_individuals)
    e = rng.normal(size=tree_sequence.num_individuals) * np.sqrt(sigma)
    b = rng.normal(size=5)
    y = X @ b + g + e
    return y, X, b, g


if __name__ == "__main__":

    fig_dir = os.path.join(os.path.dirname(__file__), "figs")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    num_threads = 4
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

    mu = 1e-10
    traits, covariates, fixef, genetic_values = simulate(1.0, 1.0, ts, mu, rng=rng)
    ols = covariates @ np.linalg.solve(covariates.T @ covariates, covariates.T @ traits)
    residuals = traits - ols

    tau_grid = np.array([0.01, 0.1, 1.0, 10., 100.])
    rank_grid = np.array([0, 1, 10, 100])
    covr = CovarianceModel(ts, mu)
    iterations = np.empty((rank_grid.size, tau_grid.size))
    for j, rank in enumerate(rank_grid):
        if rank > 0: prec = LowRankPreconditioner(covr, rank=rank, num_vectors=2*rank)
        for i, tau in enumerate(tau_grid):
            M = None if rank == 0 else lambda x: prec(1.0, tau, x)
            _, (itt, _) = covr.solve(1.0, tau, residuals, preconditioner=M, return_info=True)
            iterations[j, i] = itt

    for j, rank in enumerate(rank_grid):
        plt.plot(tau_grid, iterations[j], "-o", label=f"rank={rank}")
    plt.legend()
    plt.xscale("log")
    plt.xlabel(r"$\tau^2 / \sigma^2$")
    plt.ylabel("CG iterations")
    plt.title("CG convergence rate with\nlow rank preconditioner (1000 diploids)")
    plt.savefig("figs/check_conjugate_gradient_preconditioning.png")

