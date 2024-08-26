import numpy as np
from typing import Callable

from likelihoods import stochastic_gradient_reml
from linear_operators import TraitCovariance, NystromPreconditioner

    
def optimize_variance_components(
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
    verbose: bool = True,
    callback: Callable = None,
):
    """
    Estimate variance components by stochastic gradient descent on REML
    """

    X, _ = np.linalg.qr(X)  # orthonormalise (could use R to get back beta on correct scale)

    # AdaDelta (https://arxiv.org/pdf/1212.5701)
    state = np.array([sigma, tau])
    running_mean = state
    epsilon = np.eye(state.size) * epsilon
    numerator = np.zeros_like(epsilon)
    denominator = np.zeros_like(epsilon)
    for itt in range(maxitt):
        assert np.all(state > 0)
        gradient = -2 * stochastic_gradient_reml(*state, y, X, covariance, preconditioner, num_samples=num_samples, rng=rng)
        gradient = np.expand_dims(gradient, 1)
        denominator = (1 - decay) * denominator + decay * gradient @ gradient.T
        # full hessian approximation:
        #update = -np.linalg.cholesky(numerator + epsilon) @ \
        #    np.linalg.solve(np.linalg.cholesky(denominator + epsilon), gradient)
        update = -np.sqrt(np.diag(numerator + epsilon)).reshape(-1, 1) / \
           np.sqrt(np.diag(denominator + epsilon)).reshape(-1, 1) * gradient
        numerator = (1 - decay) * numerator + decay * update @ update.T
        state = state + update.squeeze()
        running_mean = (1 - decay) * running_mean + decay * state
        if verbose: print(f"Iteration {itt}: {state.round(2)}, {running_mean.round(2)}")
        if callback is not None: callback(running_mean)
        # TODO: could exit based on norm of numerator or denominator
    sigma, tau = running_mean

    return sigma, tau
