"""
Check that REML gradient approximation is unbiased
"""

import os
import numba
import numpy as np
import msprime

from tslmm.tslmm import tslmm
from tslmm.tslmm import _explicit_gradient, _explicit_covariance_matrix, _explicit_average_information


def simulate(sigma, tau, tree_sequence, mutation_rate, rng=None, subset=None, center_covariance=False):
    """
    `subset` is only used for centering
    """
    if rng is None: rng = np.random.default_rng()
    if subset is None: subset = np.arange(tree_sequence.num_individuals)
    G = _explicit_covariance_matrix(
        sigma, tau, tree_sequence, mutation_rate, 
        center_around=subset if center_covariance else None,
    )
    U = np.linalg.cholesky(G)
    X = rng.normal(size=(tree_sequence.num_individuals, 5))
    b = rng.normal(size=5)
    y = X @ b + U @ rng.normal(size=tree_sequence.num_individuals)
    return y, X, b


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
    subset = np.arange(ts.num_individuals)

    true_sigma, true_tau = 1.5, 0.5
    varcov = np.array([true_sigma, true_tau])
    mu = 1e-10
    traits, covariates, fixef = simulate(*varcov, ts, mu, rng=rng)

    lmm = tslmm(ts, mu, traits[subset], covariates[subset], phenotyped_individuals=subset, rng=rng)
    lmm.set_variance_components(variance_components=np.ones(2))
    direction = np.array([1.0, 0.4])
    grid = np.linspace(-1.0, 1.0, 20)
    est_average_information = np.empty((grid.size, 4))
    exp_average_information = np.empty((grid.size, 4))
    pars = np.empty((grid.size, 2))
    for i, step in enumerate(grid):
        sigma, tau = varcov + step * direction
        _, _, _, est_ai = lmm._reml_stochastic_average_information(
            sigma, tau, traits, covariates, 
            covariance=lmm.covariance, preconditioner=lmm.preconditioner, 
            trace_samples=5, rng=rng,
        )
        est_average_information[i] = est_ai.ravel()
        exp_average_information[i] = _explicit_average_information(sigma, tau, ts, mu, traits, covariates, subset=subset).ravel()
        pars[i] = sigma, tau

    import matplotlib.pyplot as plt
    labels = ['sigma - sigma', 'sigma - tau', 'tau - sigma', 'tau - tau']
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for i, ax in enumerate(axs):
        x, y = exp_average_information[:,i], est_average_information[:,i]
        ll, ul = min(x.min(), y.min()), max(x.max(), y.max())
        ax.scatter(x, y, c='red')
        ax.plot([ll,ul], [ll,ul], ls='--', color='grey')
        ax.set_xlabel("expected average information")
        ax.set_ylabel("estimated average information")
        ax.set_title(labels[i])
        ax.set_xscale('log')
        ax.set_yscale('log')

    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, "check_stochastic_average_information.png"))
