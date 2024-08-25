import numpy as np


def genetic_values(sigma, tau, residuals, covariance, preconditioner, num_samples=0, rng=None):
    """
    Calculate BLUPs, exact expectation and approximate variance.

    Individual formuation:

        y = g + e, g ~ N(0, t Q), e ~ N(0, s I)
        Q = Z (L L')^{-1} Z'
        M = Q^{-1} / t + I / s

        E[g|y] = M^{-1} y / s = (Q^{-1} / t + I / s)^{-1} y / s
               = (I - s (Q t + I s)^{-1}) y

        V[g|y] = M^{-1} = (Q^{-1} / t + I / s)^{-1}
               = I s - s^2 (t Q + I s)^{-1}

    Edge formulation:

        y = X u + e, u ~ N(0, t I), e ~ N(0, s I)
        X = Z L^{-T}

        f = C + (y - X u)' (y - X u) / s + u' u / t
          = C + u' (X' X / s + I / t) u - 2 (y' X / s) u
          = C + u' M u - 2 b' u

        M = L^{-1} Z' (I / s) Z L^{-T} + I / t
        b = L^{-1} Z' y / s

        E[u|y] = M^{-1} b ==> binomial inverse theorem ==>
               = (I t - t^2 / s^2 L^{-1} Z' (I / s + t / s^2 Z (L' L)^{-1} Z')^{-1} Z L^{-T}) b
               = (I t - t^2 L^{-1} Z' (I s + t Z (L' L)^{-1} Z')^{-1} Z L^{-T}) L^{-1} Z' y / s

        V[u|y] = M^{-1} ==> binomial inverse theorem ==>
               = I t - t^2 L^{-1} Z' (I s + t Z (L' L)^{-1} Z')^{-1} Z L^{-T} 
    """

    if rng is None: rng = np.random.default_rng()
    dim = covariance.dim
    M = lambda y: preconditioner(sigma, tau, y)

    solution, iterations, converged = covariance.solve(sigma, tau, residuals, preconditioner=M)
    assert converged

    # expected value
    Ey = residuals - sigma * solution
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
