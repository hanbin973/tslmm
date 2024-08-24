import numpy as np


def genetic_values(sigma, tau, y, covariance, preconditioner, num_samples=0, rng=None):
    """
    y = g + e, g ~ N(0, t Q), e ~ N(0, s I)

    where 

    Q = Z (L L')^{-1} Z'
    M = Q^{-1} / t + I / s

    then

    E[g|y] = M^{-1} y / s = (Q^{-1} / t + I / s)^{-1} y / s
           = (I - s (Q t + I s)^{-1}) y

    V[g|y] = M^{-1} = (Q^{-1} / t + I / s)^{-1}
           = I s - s^2 (t Q + I s)^{-1}
    """

    if rng is None: rng = np.random.default_rng()
    dim = covariance.dim
    M = lambda y: preconditioner(sigma, tau, y)

    solution, iterations, converged = covariance.solve(sigma, tau, y, preconditioner=M)
    assert converged

    # expected value
    Ey = y - sigma * solution
    Vy = np.full(dim, np.nan)

    # estimate variance with stochastic estimator
    def _covariance_diag(test_vectors):
        if test_vectors.ndim == 1: test_vectors = test_vectors.reshape(-1, 1)
        sketch, _, _ = covariance.solve(sigma, tau, test_vectors, preconditioner=M)
        sketch = sigma * (test_vectors - sigma * sketch)
        return sketch

    if num_samples > 0: 
        Vy[:] = xdiag(_covariance_diag, dim, num_samples, rng) 

    return Ey, Vy
