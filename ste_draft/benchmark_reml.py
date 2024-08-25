import time
import numba
import numpy as np
import msprime
import matplotlib.pyplot as plt

from likelihoods import exact_loglikelihood_reml, exact_gradient_reml, stochastic_gradient_reml
from linear_operators import TraitCovariance, NystromPreconditioner


def compare_exact_vs_stochastic(sigma, tau, y, X, covariance, preconditioner, rng=None, num_samples=4):
    loglik = exact_loglikelihood_reml(sigma, tau, y, X, covariance)

    st = time.time()
    exact = exact_gradient_reml(sigma, tau, y, X, covariance)
    en = time.time()
    exact_timing = en - st

    st = time.time()
    stochastic = stochastic_gradient_reml(sigma, tau, y, X, covariance, preconditioner, num_samples=num_samples, rng=rng, variance_reduction=False)
    en = time.time()
    stochastic_timing = en - st

    st = time.time()
    stochastic_vr = stochastic_gradient_reml(sigma, tau, y, X, covariance, preconditioner, num_samples=num_samples, rng=rng, variance_reduction=True)
    en = time.time()
    stochastic_vr_timing = en - st

    print(exact)
    print(stochastic)
    print(stochastic_vr)
    print(f"timings, exact: {exact_timing:.2f}, stochastic: {stochastic_timing:.2f}, stochastic-vr: {stochastic_vr_timing:.2f}", flush=True)

    return loglik, exact, stochastic, stochastic_vr


if __name__ == "__main__":
    
    num_threads = 4
    numba.set_num_threads(num_threads)
    
    n_samples = 50
    n_pops = 20
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
    
    sigma_grid = np.linspace(1.0, 2.0, 25)
    tau_grid = np.linspace(0.3, 0.8, 25)
    
    sigma_like = []
    sigma_exact = []
    sigma_approx = []
    sigma_approx_vr = []
    for sigma in sigma_grid:
        like, exact, stochastic, stochastic_vr = compare_exact_vs_stochastic(sigma, true_tau, traits, covariates, covariance, preconditioner, rng=rng)
        sigma_like.append(like)
        sigma_exact.extend(exact)
        sigma_approx.extend(stochastic)
        sigma_approx_vr.extend(stochastic_vr)
    sigma_exact = np.array(sigma_exact).reshape(-1, 2)
    sigma_approx = np.array(sigma_approx).reshape(-1, 2)
    sigma_approx_vr = np.array(sigma_approx_vr).reshape(-1, 2)
    
    tau_like = []
    tau_exact = []
    tau_approx = []
    tau_approx_vr = []
    for tau in tau_grid:
        like, exact, stochastic, stochastic_vr = compare_exact_vs_stochastic(true_sigma, tau, traits, covariates, covariance, preconditioner, rng=rng)
        tau_like.append(like)
        tau_exact.extend(exact)
        tau_approx.extend(stochastic)
        tau_approx_vr.extend(stochastic_vr)
    tau_exact = np.array(tau_exact).reshape(-1, 2)
    tau_approx = np.array(tau_approx).reshape(-1, 2)
    tau_approx_vr = np.array(tau_approx_vr).reshape(-1, 2)
    
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs[0, 0].plot(sigma_grid, sigma_exact[:, 0], label="exact")
    axs[0, 0].scatter(sigma_grid, sigma_approx[:, 0], label="approx")
    axs[0, 0].scatter(sigma_grid, sigma_approx_vr[:, 0], label="approx-vr")
    axs[0, 0].axhline(y=0.0, color='black', linestyle='--')
    axs[0, 0].axvline(x=true_sigma, color='black', linestyle='--')
    axs[0, 0].set_ylabel("gradient wrt sigma")
    axs[0, 0].set_xlabel("sigma")
    axs[0, 0].legend()
    axs[0, 1].plot(tau_grid, tau_exact[:, 1], label="exact")
    axs[0, 1].scatter(tau_grid, tau_approx[:, 1], label="approx")
    axs[0, 1].scatter(tau_grid, tau_approx_vr[:, 1], label="approx-vr")
    axs[0, 1].axhline(y=0.0, color='black', linestyle='--')
    axs[0, 1].axvline(x=true_tau, color='black', linestyle='--')
    axs[0, 1].set_ylabel("gradient wrt tau")
    axs[0, 1].set_xlabel("tau")
    axs[0, 1].legend()
    axs[1, 0].plot(sigma_grid, sigma_like)
    axs[1, 0].axvline(x=true_sigma, color='black', linestyle='--')
    axs[1, 0].set_ylabel("loglik")
    axs[1, 0].set_xlabel("sigma")
    axs[1, 1].plot(tau_grid, tau_like)
    axs[1, 1].axvline(x=true_tau, color='black', linestyle='--')
    axs[1, 1].set_ylabel("loglik")
    axs[1, 1].set_xlabel("tau")
    fig.tight_layout()

    plt.savefig("figs/stochastic_grad_reml_benchmark.png")
