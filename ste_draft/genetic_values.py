import numpy as np

from linear_operators import TraitCovariance, NystromPreconditioner
from trace_estimators import xdiag


def genetic_values(
    sigma: float, 
    tau: float, 
    residuals: np.ndarray, 
    covariance: TraitCovariance, 
    preconditioner: NystromPreconditioner,
    num_samples: int = 0, 
    rng: np.random.Generator = None,
) -> (np.ndarray, np.ndarray):
    """
    Calculate BLUPs and estimate posterior variance

    Individual formuation:

        y = g + e, g ~ N(0, t Q), e ~ N(0, s I)
        Q = Z (L L')^{-1} Z'
        M = Q^{-1} / t + I / s

        E[g|y] = M^{-1} y / s = (Q^{-1} / t + I / s)^{-1} y / s
               = (I - s (Q t + I s)^{-1}) y

        V[g|y] = M^{-1} = (Q^{-1} / t + I / s)^{-1}
               = I s - s^2 (t Q + I s)^{-1}
    """

    if rng is None: rng = np.random.default_rng()
    M = lambda y: preconditioner(sigma, tau, y)

    solution, iterations, converged = covariance.solve(sigma, tau, residuals, preconditioner=M)
    assert converged

    # genetic values
    E_g = residuals - sigma * solution
    V_g = np.full(covariance.dim, np.nan)

    def _covariance_diag(test_vectors):
        if test_vectors.ndim == 1: test_vectors = test_vectors.reshape(-1, 1)
        sketch, _, _ = covariance.solve(sigma, tau, test_vectors, preconditioner=M)
        sketch = sigma * (test_vectors - sigma * sketch)
        return sketch

    if num_samples > 0: 
        V_g[:] = xdiag(_covariance_diag, covariance.dim, num_samples, rng) 

    return E_g, V_g


def edge_effects(
    sigma: float, 
    tau: float, 
    residuals: np.ndarray, 
    covariance: TraitCovariance, 
    preconditioner: NystromPreconditioner,
    num_samples: int = 0, 
    rng: np.random.Generator = None,
) -> (np.ndarray, np.ndarray):
    """
    Calculate BLUPs and estimate posterior variance

    Edge formulation:

        y = X u + e, u ~ N(0, t I), e ~ N(0, s I)
        X = Z L^{-T}

        f = C + (y - X u)' (y - X u) / s + u' u / t
          = C + u' (X' X / s + I / t) u - 2 (y' X / s) u
          = C + u' M u - 2 b' u

        Q = Z (L L')^{-1} Z'
        M = L^{-1} Z' (I / s) Z L^{-T} + I / t
        b = L^{-1} Z' y / s

        E[u|y] = M^{-1} b ==> binomial inverse theorem ==>
               = (I t - t^2 / s^2 L^{-1} Z' (I / s + t / s^2 Z (L' L)^{-1} Z')^{-1} Z L^{-T}) b
               = (I t - t^2 L^{-1} Z' (I s + t Z (L' L)^{-1} Z')^{-1} Z L^{-T}) L^{-1} Z' y / s
               = (t/s) L^{-1} Z' (y - t (I s + t Q)^{-1} Q y)

        V[u|y] = M^{-1} ==> binomial inverse theorem ==>
               = I t - t^2 L^{-1} Z' (I s + t Z (L' L)^{-1} Z')^{-1} Z L^{-T} 
    """

    if rng is None: rng = np.random.default_rng()
    M = lambda y: preconditioner(sigma, tau, y)

    # t (s I + t Z' (L L')^{-1} Z)^{-1} Z' (L L')^{-1} Z x
    solution, iterations, converged = covariance.solve(sigma, tau, covariance(0, tau, residuals))
    assert converged

    # edge effects
    E_u = covariance._factor_adjoint(residuals - solution) * tau / sigma  # L^{-1} Z' x
    V_u = np.full(covariance.factor_dim)

    def _covariance_diag(test_vectors):
        if test_vectors.ndim == 1: test_vectors = test_vectors.reshape(-1, 1)
        sketch = covariance._factor(test_vectors) * tau
        sketch, _, _ = covariance.solve(sigma, tau, sketch, preconditioner=M)
        sketch = covariance._factor_adjoint(sketch)
        sketch = tau * (test_vectors - sketch)
        return sketch

    # expensive, requires a bunch of dense vectors of length `split_ts.num_edges`
    if num_samples > 0: 
        V_u[:] = xdiag(_covariance_diag, covariance.factor_dim, num_samples, rng) 

    return E_u, V_u
