"""
Check that out-of-sample BLUPs are reasonably close to truth when genetic component is large
"""

import os
import numba
import numpy as np
import msprime
import scipy

from tsblup.tslmm import tslmm
from tsblup.tslmm import _explicit_covariance_matrix, _explicit_posterior

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


def calculate_coverage(truth, prediction, stddev, interval_width):
    scale = abs(scipy.stats.norm.ppf((1 - interval_width) / 2))
    return np.sum(np.logical_and(truth > prediction - scale * stddev, truth < prediction + scale * stddev)) / truth.size


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
    not_subset = np.setdiff1d(np.arange(ts.num_individuals), subset)

    true_sigma, true_tau = 1.5, 5.0
    varcov = np.array([true_sigma, true_tau])
    mu = 1e-10
    traits, covariates, fixef, genetic_values = simulate(*varcov, ts, mu, rng=rng)

    num_samples=100
    lmm = tslmm(ts, mu, traits[subset], covariates[subset], phenotyped_individuals=subset, variance_components=varcov, rng=rng)
    blups, var_blups = lmm.predict(np.arange(ts.num_individuals), variance_samples=num_samples)

    _, post_cov_blups = _explicit_posterior(
        true_sigma, true_tau, ts, mu, traits, covariates, 
        subset=subset, predict_subset=np.arange(ts.num_individuals),
    )
    exact_var_blups = post_cov_blups.diagonal()

    std_dev = np.sqrt(np.abs(var_blups))
    expected_coverage = np.linspace(0.01, 0.99, 20)
    observed_coverage_subset = [calculate_coverage(genetic_values[subset], blups[subset], std_dev[subset], x) for x in expected_coverage]
    observed_coverage_not_subset = [calculate_coverage(genetic_values[not_subset], blups[not_subset], std_dev[not_subset], x) for x in expected_coverage]
    
    fig, axs = plt.subplots(3, 2, figsize=(8, 12))
    axs[0, 0].scatter(genetic_values[subset], blups[subset], c='red', s=8)
    axs[0, 0].axline((0, 0), slope=1, color='black', linestyle='--')
    axs[0, 0].set_ylabel("BLUPs")
    axs[0, 0].set_xlabel("true genetic values")
    axs[0, 0].set_title("phenotyped")
    axs[0, 1].scatter(genetic_values[not_subset], blups[not_subset], c='red', s=8)
    axs[0, 1].axline((0, 0), slope=1, color='black', linestyle='--')
    axs[0, 1].set_ylabel("BLUPs")
    axs[0, 1].set_xlabel("true genetic values")
    axs[0, 1].set_title("not phenotyped")
    axs[1, 0].scatter(exact_var_blups[subset], var_blups[subset], c='red', s=8)
    axs[1, 0].axline((0, 0), slope=1, color='black', linestyle='--')
    axs[1, 0].set_ylabel(f"approx var[BLUPs] ({num_samples} samples)")
    axs[1, 0].set_xlabel("exact var[BLUPs]")
    axs[1, 0].set_title("phenotyped")
    axs[1, 1].scatter(exact_var_blups[not_subset], var_blups[not_subset], c='red', s=8)
    axs[1, 1].axline((0, 0), slope=1, color='black', linestyle='--')
    axs[1, 1].set_ylabel(f"approx var[BLUPs] ({num_samples} samples)")
    axs[1, 1].set_xlabel("exact var[BLUPs]")
    axs[1, 1].set_title("not phenotyped")
    axs[2, 0].scatter(expected_coverage, observed_coverage_subset, c='red')
    axs[2, 0].axline((0, 0), slope=1, color='black', linestyle='--')
    axs[2, 0].set_ylabel("actual coverage")
    axs[2, 0].set_xlabel("expected coverage")
    axs[2, 0].set_title("phenotyped")
    axs[2, 1].scatter(expected_coverage, observed_coverage_not_subset, c='red')
    axs[2, 1].axline((0, 0), slope=1, color='black', linestyle='--')
    axs[2, 1].set_ylabel("actual coverage")
    axs[2, 1].set_xlabel("expected coverage")
    axs[2, 1].set_title("not phenotyped")
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, "check_genetic_values.png"))
