"""
Check that REML gradient approximation is unbiased
"""

import os
import numba
import numpy as np
import msprime

from tslmm.tslmm import tslmm
from tslmm.tslmm import _explicit_gradient, _explicit_covariance_matrix


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

    lmm = tslmm(ts, mu, traits[subset], covariates[subset], phenotyped_individuals=subset, rng=rng,)
    lmm.set_variance_components(variance_components=np.ones(2),)
    direction = np.array([1.0, 0.4])
    grid = np.linspace(-1.0, 1.0, 20)
    est_grad = np.empty((grid.size, 2))
    exp_grad = np.empty((grid.size, 2))
    pars = np.empty((grid.size, 2))
    for i, step in enumerate(grid):
        sigma, tau = varcov + step * direction
        est_grad[i], *_ = lmm._reml_stochastic_average_information(
            sigma, tau, traits, covariates, 
            covariance=lmm.covariance, preconditioner=lmm.preconditioner, 
            trace_samples=5, rng=rng,
        )
        exp_grad[i] = _explicit_gradient(sigma, tau, ts, mu, traits, covariates, subset=subset)
        pars[i] = sigma, tau

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].axhline(y=0.0, color='black', linestyle='--')
    axs[0].axvline(x=true_sigma, color='black', linestyle='--')
    axs[0].plot(pars[:, 0], exp_grad[:, 0], c='gray', label="exact")
    axs[0].scatter(pars[:, 0], est_grad[:, 0], c='red', s=8, label="approx")
    axs[0].set_ylabel("gradient wrt sigma")
    axs[0].set_xlabel("sigma")
    axs[0].legend()
    axs[1].axhline(y=0.0, color='black', linestyle='--')
    axs[1].axvline(x=true_tau, color='black', linestyle='--')
    axs[1].plot(pars[:, 1], exp_grad[:, 1], c='gray', label="exact")
    axs[1].scatter(pars[:, 1], est_grad[:, 1], c='red', s=8, label="approx")
    axs[1].set_ylabel("gradient wrt tau")
    axs[1].set_xlabel("tau")
    axs[1].legend()
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, "check_stochastic_ai_gradient.png"))
