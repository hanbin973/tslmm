"""
Check that SGD implementation gets somewhere close to truth
"""

import os
import numpy as np
import numba
import msprime

from tsblup.tslmm import tslmm
from tsblup.tslmm import _explicit_reml, _explicit_covariance_matrix

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
    return y, X


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
    subset = np.arange(ts.num_individuals)[::2]

    true_sigma, true_tau = 1.5, 0.5
    varcov = np.array([true_sigma, true_tau])
    mu = 1e-10
    traits, covariates = simulate(*varcov, ts, mu, rng=rng)
    
    lmm = tslmm(ts, mu, traits[subset], covariates[subset], phenotyped_individuals=subset, sgd_verbose=True, rng=rng)
    trajectory = lmm._optimization_trajectory

    # calculate exact objective over grid (this is painfully slow, TODO make a faster pre-factorized explicit version)
    grid_sigma = np.linspace(0.8, 2.6, 25)
    grid_tau = np.linspace(0.1, 2.0, 25)
    grid_loglik = np.zeros((grid_sigma.size, grid_tau.size))
    for i, tau in enumerate(grid_tau):
        for j, sigma in enumerate(grid_sigma):
            grid_loglik[i, j] = _explicit_reml(sigma, tau, ts, mu, traits, covariates, subset=subset)
    
    # visualise SGD trajectory on exact likelihood surface
    mesh = plt.pcolormesh(
        np.append(grid_sigma - 0.05, grid_sigma[-1] + 0.05),
        np.append(grid_tau - 0.05, grid_tau[-1] + 0.05),
        grid_loglik, 
    )
    plt.contour(grid_sigma, grid_tau, grid_loglik, colors="red")
    cbar = plt.colorbar(mesh)
    cbar.set_label("REML")
    for (xm, ym), (x, y) in zip(trajectory[:-1], trajectory[1:]):
        plt.plot((xm, x), (ym, y), '-b')
    plt.plot(true_sigma, true_tau, 'xr')
    plt.xlabel("$\\sigma^2$")
    plt.ylabel("$\\tau^2$")
    plt.title(f"SGD on REML\n{ts.num_individuals} diploids, {lmm.covariance.factor_dim} edges, {covariates.shape[1]} covariates")
    plt.savefig("figs/check_gradient_descent_2.png")
    plt.clf()
