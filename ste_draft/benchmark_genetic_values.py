import time
import numba
import numpy as np
import msprime
import matplotlib.pyplot as plt
import scipy

from likelihoods import genetic_values
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
    covariance = TraitCovariance(ts, mutation_rate=1e-8)
    preconditioner = NystromPreconditioner(covariance, rank=100, samples=200, seed=1)

    true_tau = 0.1
    true_sigma = 0.5
    rng = np.random.default_rng(1024)
    _, true_genetic_values, traits = covariance.simulate(true_sigma, true_tau, rng)

    predictions, var_predictions = genetic_values(true_sigma, true_tau, traits, covariance, preconditioner, diag_samples=10)
    std_dev = np.sqrt(var_predictions)
    
    plt.scatter(true_genetic_values, predictions, s=4)
    plt.axline((0, 0), slope=1, linestyle="--", color="black")
    plt.xlabel("True genetic values")
    plt.ylabel("Fitted genetic values")
    plt.savefig("figs/genetic_values_benchmark.png")
    plt.clf()

    # coverage
    def measure_coverage(truth, prediction, stddev, interval_width):
        scale = abs(scipy.stats.norm.ppf((1 - interval_width) / 2))
        return np.sum(np.logical_and(truth > prediction - scale * stddev, truth < prediction + scale * stddev)) / truth.size

    expected_coverage = np.linspace(0.01, 0.99, 20)
    observed_coverage = [measure_coverage(true_genetic_values, predictions, std_dev, x) for x in expected_coverage]
    plt.scatter(expected_coverage, observed_coverage, s=4, color="red")
    plt.axline((0, 0), slope=1, linestyle="--", color="black")
    plt.xlabel("Expected coverage")
    plt.ylabel("Observed coverage")
    plt.savefig("figs/genetic_values_var.png")
    plt.clf()
