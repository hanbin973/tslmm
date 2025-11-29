import numpy as np
import numba
import msprime

from likelihoods import exact_loglikelihood_reml
from linear_operators import TraitCovariance, NystromPreconditioner
from optimization import optimize_variance_components

import matplotlib.pyplot as plt

if __name__ == "__main__":
    num_threads = 5
    numba.set_num_threads(num_threads)
    rng = np.random.default_rng(1)
    
    n_samples = 50
    n_pops = 10
    s_length = 1e5
    ne = 1e4
    island_model = msprime.Demography.island_model([ne] * n_pops, 1e-5 / n_pops)
    for i in range(1, n_pops):
        island_model.add_mass_migration(time=2*ne, source=i, dest=0, proportion=1.0)
    ts = msprime.sim_ancestry(
        samples={f"pop_{i}": n_samples for i in range(n_pops)},
        recombination_rate=1e-8,
        sequence_length=s_length,
        demography=island_model,
        random_seed=1024,
    )
    
    covariance = TraitCovariance(ts, mutation_rate=1e-10)
    preconditioner = NystromPreconditioner(covariance, rank=100, samples=200, seed=1)
    
    true_tau = 0.5
    true_sigma = 1.5
    rng = np.random.default_rng(1024)
    _, _, random_effects = covariance.simulate(true_sigma, true_tau, rng)
    covariates = np.random.randn(ts.num_individuals, 5)
    true_beta = np.random.randn(5)
    traits = covariates @ true_beta + random_effects
    
    # optimize, store SGD trajectory
    epsilon = 1e-3  # acts like learning rate (regularises hessian inverse)
    decay = 0.3  # forgetting rate for hessian approximation / running average
    trajectory = []
    optimize_variance_components(2.6, 0.2, traits, covariates, covariance, preconditioner, epsilon=epsilon, decay=decay, num_samples=5, rng=rng, callback=lambda x: trajectory.append(x))

    # visualise error vs iteration count
    truth = np.array([true_sigma, true_tau])
    err = np.array([np.linalg.norm(x - truth) for x in trajectory])
    plt.plot(np.arange(err.size), err, "-r")
    plt.xlabel("Iteration")
    plt.ylabel("$|\\hat{x} - x|_2$")
    plt.title(f"SGD (AdaDelta) convergence\nepsilon: {epsilon}, decay: {decay}")
    plt.savefig("figs/optimization_reml_convergence.png")
    plt.clf()

    # calculate exact objective over grid
    grid_sigma = np.linspace(0.8, 2.6, 25)
    grid_tau = np.linspace(0.2, 2.0, 25)
    grid_loglik = np.zeros((grid_sigma.size, grid_tau.size))
    for i, tau in enumerate(grid_tau):
        for j, sigma in enumerate(grid_sigma):
            grid_loglik[i, j] = exact_loglikelihood_reml(sigma, tau, traits, covariates, covariance, use_qr=True)
    
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
    plt.title(f"SGD on REML\n{covariance.dim} diploids, {covariance.factor_dim} edges, {covariates.shape[1]} covariates")
    plt.savefig("figs/optimization_reml.png")
    plt.clf()
