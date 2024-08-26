import numpy as np
import numba
import msprime

from likelihoods import stochastic_gradient_reml, exact_loglikelihood_reml
from linear_operators import TraitCovariance, NystromPreconditioner

num_threads = 4
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
preconditioner = NystromPreconditioner(covariance, rank=10, samples=200, seed=1)

true_tau = 0.5
true_sigma = 1.5
rng = np.random.default_rng(1024)
_, _, random_effects = covariance.simulate(true_sigma, true_tau, rng)
covariates = np.random.randn(ts.num_individuals, 5)
true_beta = np.random.randn(5)
traits = covariates @ true_beta + random_effects
    

def estimate_variance_components(
    sigma: float, 
    tau: float, 
    y: np.ndarray, 
    X: np.ndarray, 
    covariance: TraitCovariance,
    preconditioner: NystromPreconditioner,
    rng: np.random.Generator = None, 
    num_samples: int = 10, 
    decay: float = 0.3, 
    epsilon: float = 1e-3, 
    maxitt: int = 100,
    callback=None,
):

    X, _ = np.linalg.qr(X)  # orthonormalise (could use R to get back beta on correct scale)

    # AdaDelta (https://arxiv.org/pdf/1212.5701) but using full Hessian approximation
    params = np.array([sigma, tau])
    regularise = np.eye(params.size) * epsilon
    running_average = params
    numerator = np.zeros_like(regularise)
    denominator = np.zeros_like(regularise)
    for itt in range(maxitt):
        print(f"Iteration {itt}, {params.round(2)}, {running_average.round(2)}")
        assert np.all(params > 0)
        gradient = -2 * stochastic_gradient_reml(*params, y, X, covariance, preconditioner, num_samples=num_samples, rng=rng)
        gradient = np.expand_dims(gradient, 1)
        denominator = (1 - decay) * denominator + decay * gradient @ gradient.T
        update = -np.linalg.cholesky(numerator + regularise) @ \
            np.linalg.solve(np.linalg.cholesky(denominator + regularise), gradient)
        numerator = (1 - decay) * numerator + decay * update @ update.T
        params = params + update.squeeze()
        running_average = (1 - decay) * running_average + decay * params
        if callback is not None: callback(running_average)
        # could exit based on norm of numerator or denominator
    sigma, tau = running_average

    return sigma, tau



grid_sigma = np.linspace(0.8, 2.6, 25)
grid_tau = np.linspace(0.2, 2.0, 25)
grid_loglik = np.zeros((grid_sigma.size, grid_tau.size))
for i, tau in enumerate(grid_tau):
    for j, sigma in enumerate(grid_sigma):
        grid_loglik[i, j] = exact_loglikelihood_reml(sigma, tau, traits, covariates, covariance, use_qr=True)

trajectory = []
estimate_variance_components(2.0, 0.2, traits, covariates, covariance, preconditioner, num_samples=5, rng=rng, callback=lambda x: trajectory.append(x))

import matplotlib.pyplot as plt
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
