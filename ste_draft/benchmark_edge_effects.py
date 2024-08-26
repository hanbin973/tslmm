import time
import numba
import numpy as np
import msprime
import matplotlib.pyplot as plt
import scipy

from genetic_values import edge_effects
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
    true_edge_effects, _, traits = covariance.simulate(true_sigma, true_tau, rng)

    predictions, var_predictions = edge_effects(true_sigma, true_tau, traits, covariance, preconditioner, rng=rng, num_samples=0)
    std_dev = np.sqrt(var_predictions)
    
    plt.scatter(true_edge_effects, predictions, s=4)
    plt.axline((0, 0), slope=1, linestyle="--", color="black")
    plt.text(0.01, 0.99, f"$r^2 = {np.corrcoef(true_edge_effects, predictions)[0, 1] ** 2:.2f}$", transform=plt.gca().transAxes, ha='left', va='top')
    plt.ylabel("Fitted edge effects")
    plt.xlabel("True edge effects")
    plt.savefig("figs/edge_effects_benchmark.png")
    plt.clf()

    ## variance
    #plt.scatter(np.sqrt(exact_post_variance.diagonal()), std_dev, s=4)
    #plt.axline((0, 0), slope=1, linestyle="--", color="black")
    #plt.text(0.01, 0.99, f"$r^2 = {np.corrcoef(np.sqrt(exact_post_variance.diagonal()), std_dev)[0, 1] ** 2:.2f}$", transform=plt.gca().transAxes, ha='left', va='top')
    #plt.ylabel("estimated Sd[g|y]")
    #plt.xlabel("true Sd[g|y]")
    #plt.tight_layout()
    #plt.savefig("figs/genetic_values_var.png")
    #plt.clf()

    ## coverage
    #def measure_coverage(truth, prediction, stddev, interval_width):
    #    scale = abs(scipy.stats.norm.ppf((1 - interval_width) / 2))
    #    return np.sum(np.logical_and(truth > prediction - scale * stddev, truth < prediction + scale * stddev)) / truth.size

    #expected_coverage = np.linspace(0.01, 0.99, 20)
    #observed_coverage = [measure_coverage(true_genetic_values, predictions, std_dev, x) for x in expected_coverage]
    #plt.figure(figsize=(5, 4))
    #plt.scatter(expected_coverage, observed_coverage, s=4, color="red")
    #plt.axline((0, 0), slope=1, linestyle="--", color="black")
    #plt.xlabel("Expected coverage")
    #plt.ylabel("Observed coverage")
    #plt.tight_layout()
    #plt.savefig("figs/genetic_values_coverage.png")
    #plt.clf()
