"""
Check that SGD implementation gets somewhere close to truth
"""

import os
import numpy as np
import numba
import msprime

from tslmm.tslmm import tslmm
from tslmm.tslmm import _explicit_reml, _explicit_covariance_matrix

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
    b = rng.normal(size=X.shape[1])
    y = X @ b + g + e
    return y, X


if __name__ == "__main__":

    fig_dir = os.path.join(os.path.dirname(__file__), "figs")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    num_threads = 16
    numba.set_num_threads(num_threads)

    num_sims, num_cols = 9, 3
    num_rows = int(num_sims / num_cols)
    len_unit = 3
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(len_unit * num_cols, len_unit * num_rows))
    for i_sim in range(num_sims):
        rng = np.random.default_rng(seed=(i_sim+1))
        
        n_samples = 1000
        n_pops = 10
        s_length = 1e4
        ne = np.linspace(1e3, 1e5, n_pops)
        island_model = msprime.Demography.island_model(ne, 1e-5 / n_pops)
        for i in range(1, n_pops):
            island_model.add_mass_migration(time=2*np.max(ne), source=i, dest=0, proportion=1.0)
        ts = msprime.sim_ancestry(
            samples={f"pop_{i}": n_samples // n_pops for i in range(n_pops)},
            recombination_rate=1e-8,
            sequence_length=s_length,
            demography=island_model,
            random_seed=(i_sim+1),
        )
        subset = np.arange(ts.num_individuals)[::2]

        true_sigma, true_tau = 1.5, 0.5
        varcov = np.array([true_sigma, true_tau])
        mu = 1e-10
        traits, covariates = simulate(*varcov, ts, mu, rng=rng)
        
        lmm = tslmm(ts, mu, traits[subset], covariates[subset], phenotyped_individuals=subset, rng=rng)
        lmm.fit_variance_components(method='adadelta', haseman_elston=False, verbose=True,)
        trajectory = lmm._optimization_trajectory
        lmm_he = tslmm(ts, mu, traits[subset], covariates[subset], phenotyped_individuals=subset, rng=rng)
        lmm_he.fit_variance_components(method='adadelta', haseman_elston=True, verbose=True,)
        trajectory_he = lmm_he._optimization_trajectory

        """
        # calculate exact objective over grid (this is painfully slow, TODO make a faster pre-factorized explicit version)
        grid_sigma = np.linspace(0.8, 2.6, 25)
        grid_tau = np.linspace(0.1, 2.0, 25)
        grid_loglik = np.zeros((grid_sigma.size, grid_tau.size))
        for i, tau in enumerate(grid_tau):
            for j, sigma in enumerate(grid_sigma):
                grid_loglik[i, j] = _explicit_reml(sigma, tau, ts, mu, traits, covariates, subset=subset)
        
        """
        # visualise SGD trajectory on exact likelihood surface
        """
        mesh = plt.pcolormesh(
            np.append(grid_sigma - 0.05, grid_sigma[-1] + 0.05),
            np.append(grid_tau - 0.05, grid_tau[-1] + 0.05),
            grid_loglik, 
        )
        plt.contour(grid_sigma, grid_tau, grid_loglik, colors="red")
        cbar = plt.colorbar(mesh)
        cbar.set_label("REML")
        """

        x_idx, y_idx = i_sim % num_cols, int(i_sim / num_cols)
        for (xm, ym), (x, y) in zip(trajectory[:-1], trajectory[1:]):
            ax[y_idx, x_idx].plot((xm, x), (ym, y), '-b')
        for (xm, ym), (x, y) in zip(trajectory_he[:-1], trajectory_he[1:]):
            ax[y_idx, x_idx].plot((xm, x), (ym, y), '-r')
        ax[y_idx, x_idx].plot(true_sigma, true_tau, 'xr')
        ax[y_idx, x_idx].set_xlabel("$\\sigma^2$")
        ax[y_idx, x_idx].set_ylabel("$\\tau^2$")
        ax[y_idx, x_idx].set_title("Trial %d" % (i_sim+1))
    
    from matplotlib.lines import Line2D
    bline = Line2D([0], [0], label='Basic initialization', color='blue')
    rline = Line2D([0], [0], label='HE initialization', color='red')
    fig.legend(handles=[bline, rline], loc = 'upper right', frameon=False)

    plt.suptitle(f"SGD on REML\n{ts.num_individuals} diploids, {covariates.shape[1]} covariates")
    plt.tight_layout()
    plt.savefig("figs/check_haseman_elston.png")
    plt.clf()
