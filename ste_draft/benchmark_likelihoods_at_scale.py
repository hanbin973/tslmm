import time
import numba
import numpy as np
import msprime
import matplotlib.pyplot as plt

from likelihoods import exact_loglikelihood, exact_gradient, stochastic_gradient
from linear_operators import TraitCovariance, NystromPreconditioner


def compare_stochastic(sigma, tau, y, covariance, preconditioner, rng=None, num_samples=4):
    st = time.time()
    stochastic = stochastic_gradient(sigma, tau, y, covariance, preconditioner, num_samples=num_samples, rng=rng, variance_reduction=False)
    en = time.time()
    stochastic_timing = en - st

    st = time.time()
    stochastic_vr = stochastic_gradient(sigma, tau, y, covariance, preconditioner, num_samples=num_samples, rng=rng, variance_reduction=True)
    en = time.time()
    stochastic_vr_timing = en - st

    print(f"timings, stochastic: {stochastic_timing:.2f}, stochastic-vr: {stochastic_vr_timing:.2f}", flush=True)

    return stochastic, stochastic_vr


if __name__ == "__main__":
    
    num_threads = 4
    numba.set_num_threads(num_threads)
    
    st = time.time()
    ts = msprime.sim_ancestry(
        samples=100000,
        recombination_rate=1e-8,
        sequence_length=5e7,
        population_size=1e4,
        random_seed=1024,
    )
    covariance = TraitCovariance(ts, mutation_rate=1e-10)
    preconditioner = NystromPreconditioner(covariance, rank=500, samples=500, seed=1)
    true_tau = 1.0
    true_sigma = 1.5
    rng = np.random.default_rng(1024)
    _, _, traits = covariance.simulate(true_sigma, true_tau, rng)
    en = time.time()
    print(f"Setup time: {en - st:.2f}", flush=True)
    
    sigma_grid = np.linspace(0.1, 4.0, 20)
    tau_grid = np.linspace(0.1, 2.0, 20)
    
    tau_approx = []
    tau_approx_vr = []
    for tau in tau_grid:
        stochastic, stochastic_vr = compare_stochastic(true_sigma, tau, traits, covariance, preconditioner, rng=rng)
        tau_approx.append(np.sqrt(np.sum(stochastic ** 2)))
        tau_approx_vr.append(np.sqrt(np.sum(stochastic_vr ** 2)))
    tau_approx = np.array(tau_approx)
    tau_approx_vr = np.array(tau_approx_vr)
    tau_min = np.argmin(np.abs(tau_approx))
    
    sigma_approx = []
    sigma_approx_vr = []
    for sigma in sigma_grid:
        stochastic, stochastic_vr = compare_stochastic(sigma, true_tau, traits, covariance, preconditioner, rng=rng)
        sigma_approx.append(np.sqrt(np.sum(stochastic ** 2)))
        sigma_approx_vr.append(np.sqrt(np.sum(stochastic_vr ** 2)))
    sigma_approx = np.array(sigma_approx)
    sigma_approx_vr = np.array(sigma_approx_vr)
    sigma_min = np.argmin(np.abs(sigma_approx))
    
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    axs[0].scatter(sigma_grid, sigma_approx, label="approx")
    axs[0].scatter(sigma_grid, sigma_approx_vr, label="approx-vr")
    axs[0].axvline(x=true_sigma, color='black', linestyle='--')
    axs[0].axvline(x=sigma_grid[sigma_min], color='red', linestyle='--')
    axs[0].set_ylabel("gradient wrt sigma")
    axs[0].set_xlabel("sigma")
    axs[0].legend()
    axs[1].scatter(tau_grid, tau_approx, label="approx")
    axs[1].scatter(tau_grid, tau_approx_vr, label="approx-vr")
    axs[1].axvline(x=true_tau, color='black', linestyle='--')
    axs[1].axvline(x=tau_grid[tau_min], color='red', linestyle='--')
    axs[1].set_ylabel("gradient wrt tau")
    axs[1].set_xlabel("tau")
    axs[1].legend()
    fig.tight_layout()

    plt.savefig("figs/stochastic_grad_at_scale_benchmark.png")
