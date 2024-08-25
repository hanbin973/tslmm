"""
Tests for "linear_operators.py" and "likelihoods.py"
"""

import time
import msprime
import numba
import numpy as np
import numdifftools as nd
import scipy.sparse as sparse

from linear_operators import TraitCovariance, NystromPreconditioner
from likelihoods import exact_loglikelihood, exact_gradient, stochastic_gradient, exact_loglikelihood_reml, exact_gradient_reml, stochastic_gradient_reml


def _verify_preconditioner(sigma, tau, preconditioner, seed):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=preconditioner.dim)
    explicit_preconditioner = preconditioner.as_matrix(sigma, tau)

    # check solve(M, y)
    ck_product = np.linalg.solve(explicit_preconditioner, y)
    np.testing.assert_allclose(ck_product, preconditioner(sigma, tau, y))

    # check logdet(M)
    ck_sign, ck_logdet = np.linalg.slogdet(explicit_preconditioner)
    assert ck_sign > 0
    np.testing.assert_allclose(ck_logdet, preconditioner.logdet(sigma, tau))

    # check trace(solve(M, dM/ds))
    jac_sigma = nd.Jacobian(lambda sigma: preconditioner.as_matrix(sigma, tau))(sigma)
    jac_sigma = jac_sigma.reshape(-1, preconditioner.dim)
    ck_grad_sigma = np.linalg.solve(explicit_preconditioner, jac_sigma)
    ck_grad_sigma = np.sum(ck_grad_sigma.diagonal())
    np.testing.assert_allclose(ck_grad_sigma, preconditioner.grad_sigma(sigma, tau))

    # check trace(solve(M, dM/dt))
    jac_tau = nd.Jacobian(lambda tau: preconditioner.as_matrix(sigma, tau))(tau)
    jac_tau = jac_tau.reshape(-1, preconditioner.dim)
    ck_grad_tau = np.linalg.solve(explicit_preconditioner, jac_tau)
    ck_grad_tau = np.sum(ck_grad_tau.diagonal())
    np.testing.assert_allclose(ck_grad_tau, preconditioner.grad_tau(sigma, tau))


def _verify_covariance(sigma, tau, covariance, seed):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=covariance.dim)
    Y = rng.normal(size=(covariance.dim, 4))
    explicit_covariance = covariance.as_matrix(sigma, tau)

    # check matrix
    assert covariance.L.has_sorted_indices
    ck_genetic_matrix = covariance.Z @ sparse.linalg.spsolve(covariance.L @ covariance.L.T, covariance.Z.T.tocsc())
    ck_genetic_matrix = np.asarray(ck_genetic_matrix.todense())
    ck_matrix = np.asarray(ck_genetic_matrix * tau + np.eye(covariance.dim) * sigma)
    np.testing.assert_allclose(ck_matrix, covariance.as_matrix(sigma, tau))

    # check matvec
    ck_product = explicit_covariance @ y
    np.testing.assert_allclose(ck_product, covariance(sigma, tau, y))

    # check matvec with multiple rhs
    ck_product = explicit_covariance @ Y
    np.testing.assert_allclose(ck_product, covariance(sigma, tau, Y))

    # check inverse matvec
    ck_inverse = np.linalg.solve(explicit_covariance, y)
    inverse, iter_unconditioned, converged = covariance.solve(sigma, tau, y, rtol=1e-8)
    assert converged
    np.testing.assert_allclose(ck_inverse, inverse, rtol=1e-5)

    # check inverse matvec with preconditioner
    preconditioner = NystromPreconditioner(covariance, rank=10)
    M = lambda x: preconditioner(sigma, tau, x)
    inverse, iter_preconditioned, converged = covariance.solve(sigma, tau, y, preconditioner=M, rtol=1e-8)
    assert converged
    assert iter_preconditioned < iter_unconditioned
    np.testing.assert_allclose(ck_inverse, inverse, rtol=1e-5)

    # check inverse matvec with multiple rhs
    ck_inverse = np.linalg.solve(explicit_covariance, Y)
    inverse, _, converged = covariance.solve(sigma, tau, Y, preconditioner=M, rtol=1e-8)
    assert converged
    np.testing.assert_allclose(ck_inverse, inverse, rtol=1e-5)

    # check simulated observations
    num_reps = 1000
    simulated = [covariance.simulate(sigma, tau, rng) for _ in range(num_reps)]
    simulated_y = np.hstack([y.reshape(-1, 1) for _, _, y in simulated])
    std_dev_y = np.std(simulated_y, axis=1)
    ck_std_dev_y = np.sqrt(np.asarray(ck_matrix.diagonal()))
    #assert np.corrcoef(ck_std_dev_y, std_dev_y)[0, 1] > 0.95  # TODO: what is going wrong here?
    np.testing.assert_allclose(ck_std_dev_y, std_dev_y, rtol=1e-1)

    # check simulated genetic values
    simulated_g = np.hstack([g.reshape(-1, 1) for _, g, _ in simulated])
    std_dev_g = np.std(simulated_g, axis=1)
    ck_std_dev_g = np.sqrt(np.asarray(ck_genetic_matrix.diagonal()) * tau)
    assert np.corrcoef(ck_std_dev_g, std_dev_g)[0, 1] > 0.95
    np.testing.assert_allclose(ck_std_dev_g, std_dev_g, rtol=1e-1)

    # TODO check simulated edge effects


def _verify_gradients(sigma, tau, covariance, preconditioner, seed):
    rng = np.random.default_rng(seed)
    num_covariates = 4
    y = rng.normal(size=covariance.dim)
    X = rng.normal(size=(covariance.dim, num_covariates))

    sigma_grad = nd.Derivative(lambda sigma: exact_loglikelihood(sigma, tau, y, covariance), n=1, step=1e-4)
    tau_grad = nd.Derivative(lambda tau: exact_loglikelihood(sigma, tau, y, covariance), n=1, step=1e-4)
    ck_score = np.array([sigma_grad(sigma), tau_grad(tau)])
    np.testing.assert_allclose(ck_score, exact_gradient(sigma, tau, y, covariance))
    print(ck_score, stochastic_gradient(sigma, tau, y, covariance, preconditioner, rng=rng, num_samples=100))
    # TODO samples > rank errors out of STE

    sigma_grad_reml = nd.Derivative(lambda sigma: exact_loglikelihood_reml(sigma, tau, y, X, covariance), n=1, step=1e-4)
    tau_grad_reml = nd.Derivative(lambda tau: exact_loglikelihood_reml(sigma, tau, y, X, covariance), n=1, step=1e-4)
    ck_score = np.array([sigma_grad_reml(sigma), tau_grad_reml(tau)])
    np.testing.assert_allclose(ck_score, exact_gradient_reml(sigma, tau, y, X, covariance))
    print(ck_score, stochastic_gradient_reml(sigma, tau, y, X, covariance, preconditioner, rng=rng, num_samples=100))

    np.testing.assert_allclose(
        exact_loglikelihood_reml(sigma, tau, y, X, covariance, use_qr=False),
        exact_loglikelihood_reml(sigma, tau, y, X, covariance, use_qr=True),
    )
    np.testing.assert_allclose(
        exact_gradient_reml(sigma, tau, y, X, covariance, use_qr=False),
        exact_gradient_reml(sigma, tau, y, X, covariance, use_qr=True),
    )
    np.testing.assert_allclose(
        stochastic_gradient_reml(sigma, tau, y, X, covariance, preconditioner, rng=np.random.default_rng(1), num_samples=10),
        stochastic_gradient_reml(sigma, tau, y, np.linalg.qr(X).Q, covariance, preconditioner, rng=np.random.default_rng(1), num_samples=10),
        rtol=1e-5
    )


# -------------- #

if __name__ == "__main__":
    numba.set_num_threads(4)
    ts = msprime.sim_ancestry(
        samples=100,
        recombination_rate=1e-8,
        sequence_length=1e5,
        population_size=1e4,
        random_seed=1024,
    )
    covariance = TraitCovariance(ts, mutation_rate=1e-10)
    preconditioner = NystromPreconditioner(covariance, rank=10, samples=20, seed=1)
    _verify_preconditioner(0.9, 0.25, preconditioner, 1024)
    _verify_covariance(0.9, 0.25, covariance, 1024)
    _verify_gradients(0.9, 0.25, covariance, preconditioner, 1024)

