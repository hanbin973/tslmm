import time
import numba
import numpy as np
import msprime
import matplotlib.pyplot as plt
import scipy

from genetic_values import genetic_values
from linear_operators import TraitCovariance, NystromPreconditioner


if __name__ == "__main__":
    
    num_threads = 4
    numba.set_num_threads(num_threads)
    
    ts = msprime.sim_ancestry(
        samples=1000,
        recombination_rate=1e-8,
        sequence_length=1e6,
        population_size=1e4,
        random_seed=1024,
    )
    covariance = TraitCovariance(ts, mutation_rate=1e-10)
    preconditioner = NystromPreconditioner(covariance, rank=100, samples=200, seed=1)

    true_tau = 0.5
    true_sigma = 4.0
    rng = np.random.default_rng(1024)
    _, true_genetic_values, traits = covariance.simulate(true_sigma, true_tau, rng)
    cov_mat = covariance.as_matrix(0, 1)
    max_cov = cov_mat.diagonal().max() * true_tau + true_sigma
    exact_post_variance = np.linalg.inv(np.linalg.inv(cov_mat) / true_tau + np.eye(cov_mat.shape[0]) / true_sigma)
    print("max genetic variance / total variance:", (max_cov - true_sigma) / max_cov)

    predictions, var_predictions = genetic_values(true_sigma, true_tau, traits, covariance, preconditioner, rng=rng, num_samples=300)
    std_dev = np.sqrt(var_predictions)
    
    fig, axs = plt.subplots(2, 1, figsize=(4, 8))
    axs[0].scatter(true_genetic_values, predictions, s=4)
    #axs[0].scatter(exact_post_variance @ traits / true_sigma, predictions, s=4) <<--- include in tests
    axs[0].axline((0, 0), slope=1, linestyle="--", color="black")
    axs[0].text(0.01, 0.99, f"$r^2 = {np.corrcoef(true_genetic_values, predictions)[0, 1] ** 2:.2f}$", transform=axs[0].transAxes, ha='left', va='top')
    axs[0].set_ylabel("Fitted genetic values")
    axs[1].scatter(true_genetic_values, traits, s=4)
    axs[1].axline((0, 0), slope=1, linestyle="--", color="black")
    axs[1].text(0.01, 0.99, f"$r^2 = {np.corrcoef(true_genetic_values, traits)[0, 1] ** 2:.2f}$", transform=axs[1].transAxes, ha='left', va='top')
    axs[1].set_ylabel("Observed values")
    fig.supxlabel("True genetic values")
    plt.savefig("figs/genetic_values_benchmark.png")
    plt.clf()

    # variance
    # DEBUG <<- move to tests
    #cov_mat = covariance.as_matrix(0, 1)
    #exact_post = np.linalg.inv(np.linalg.inv(cov_mat) / true_tau + np.eye(cov_mat.shape[0]) / true_sigma)
    #prec_mat = np.linalg.inv(cov_mat * true_tau / true_sigma + np.eye(cov_mat.shape[0]))
    #post_var_2 = true_sigma * (np.eye(cov_mat.shape[0]) - prec_mat)
    #np.testing.assert_allclose(post_var_2, exact_post, rtol=1e-5)

    #from trace_estimators import xdiag
    #std_dev_2 = np.sqrt(xdiag(post_var_2, 300))
    # DEBUG

    plt.figure(figsize=(5, 4))
    plt.scatter(np.sqrt(exact_post_variance.diagonal()), std_dev, s=4)
    plt.axline((0, 0), slope=1, linestyle="--", color="black")
    plt.text(0.01, 0.99, f"$r^2 = {np.corrcoef(np.sqrt(exact_post_variance.diagonal()), std_dev)[0, 1] ** 2:.2f}$", transform=plt.gca().transAxes, ha='left', va='top')
    plt.ylabel("estimated Sd[g|y]")
    plt.xlabel("true Sd[g|y]")
    plt.tight_layout()
    plt.savefig("figs/genetic_values_var.png")
    plt.clf()

    # coverage
    def measure_coverage(truth, prediction, stddev, interval_width):
        scale = abs(scipy.stats.norm.ppf((1 - interval_width) / 2))
        return np.sum(np.logical_and(truth > prediction - scale * stddev, truth < prediction + scale * stddev)) / truth.size

    expected_coverage = np.linspace(0.01, 0.99, 20)
    observed_coverage = [measure_coverage(true_genetic_values, predictions, std_dev, x) for x in expected_coverage]
    plt.figure(figsize=(5, 4))
    plt.scatter(expected_coverage, observed_coverage, s=4, color="red")
    plt.axline((0, 0), slope=1, linestyle="--", color="black")
    plt.xlabel("Expected coverage")
    plt.ylabel("Observed coverage")
    plt.tight_layout()
    plt.savefig("figs/genetic_values_coverage.png")
    plt.clf()
