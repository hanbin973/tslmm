"""
Test equivalence of matrix operations involving covariance matrix
"""

import pytest
import numpy as np
import msprime
import scipy
import numdifftools as nd

from tslmm.tslmm import tslmm
from tslmm.tslmm import CovarianceModel, LowRankPreconditioner
from tslmm.tslmm import _explicit_covariance_matrix, _explicit_reml, _explicit_gradient


class TestCovarianceModel:
    @staticmethod
    def example(sigma, tau):
        mu = 1e-10
        ts = msprime.sim_ancestry(100, sequence_length=1e5, recombination_rate=1e-8, population_size=1e4, random_seed=1024)
        explicit_covariance = _explicit_covariance_matrix(sigma, tau, ts, mu)
        covariance = CovarianceModel(ts, mu)
        return explicit_covariance, covariance

    def test_product(self):
        rng = np.random.default_rng(1)
        sigma = 1.5
        tau = 0.5
        expl_cov, cov = self.example(sigma, tau)
        Y = rng.normal(size=(cov.dim, 4))
        X = cov(sigma, tau, Y)
        ck_X = expl_cov @ Y
        np.testing.assert_allclose(X, ck_X)

    def test_product_subset(self):
        rng = np.random.default_rng(1)
        sigma = 1.5
        tau = 0.5
        expl_cov, cov = self.example(sigma, tau)
        subset_1 = np.arange(cov.dim)[::2]
        subset_2 = np.arange(cov.dim)[::4]
        Y = rng.normal(size=(subset_2.size, 4))
        X = cov(sigma, tau, Y, rows=subset_1, cols=subset_2)
        ck_X = expl_cov[np.ix_(subset_1, subset_2)] @ Y
        np.testing.assert_allclose(X, ck_X)

    def test_solve(self):
        rng = np.random.default_rng(1)
        sigma = 1.5
        tau = 0.5
        expl_cov, cov = self.example(sigma, tau)
        Y = rng.normal(size=(cov.dim, 4))
        X = cov.solve(sigma, tau, Y)
        ck_X = np.linalg.solve(expl_cov, Y)
        np.testing.assert_allclose(X, ck_X, atol=1e-5)

    def test_solve_subset(self):
        rng = np.random.default_rng(1)
        sigma = 1.5
        tau = 0.5
        expl_cov, cov = self.example(sigma, tau)
        subset = np.arange(cov.dim)[::2]
        Y = rng.normal(size=(subset.size, 4))
        X = cov.solve(sigma, tau, Y, indices=subset)
        ck_X = np.linalg.solve(expl_cov[np.ix_(subset, subset)], Y)
        np.testing.assert_allclose(X, ck_X, atol=1e-5)

    def test_solve_preconditioned(self):
        rng = np.random.default_rng(1)
        sigma = 1.5
        tau = 0.5
        expl_cov, cov = self.example(sigma, tau)
        precond = LowRankPreconditioner(cov, rank=10, num_vectors=100)
        Y = rng.normal(size=(cov.dim, 4))
        X, (iterations, converged) = \
            cov.solve(sigma, tau, Y, preconditioner=lambda x: precond(sigma, tau, x), return_info=True)
        assert converged
        ck_X, (ck_iterations, ck_converged) = cov.solve(sigma, tau, Y, return_info=True)
        assert ck_converged
        assert iterations < ck_iterations
        np.testing.assert_allclose(X, ck_X, atol=1e-5)

    def test_solve_preconditioned_subset(self):
        rng = np.random.default_rng(1)
        sigma = 1.5
        tau = 0.5
        expl_cov, cov = self.example(sigma, tau)
        subset = np.arange(cov.dim)[::2]
        precond = LowRankPreconditioner(cov, rank=10, num_vectors=100, indices=subset)
        Y = rng.normal(size=(subset.size, 4))
        X, (iterations, converged) = \
            cov.solve(sigma, tau, Y, indices=subset, preconditioner=lambda x: precond(sigma, tau, x), return_info=True)
        assert converged
        ck_X, (ck_iterations, ck_converged) = cov.solve(sigma, tau, Y, indices=subset, return_info=True)
        assert ck_converged
        assert iterations < ck_iterations
        np.testing.assert_allclose(X, ck_X, atol=1e-5)


class TestExplicitGradient:
    def test_gradients(self):
        rng = np.random.default_rng(1)
        sigma = 1.5
        tau = 0.5
        mu = 1e-10

        ts = msprime.sim_ancestry(100, sequence_length=1e5, recombination_rate=1e-8, population_size=1e4, random_seed=1024)
        y = rng.normal(size=ts.num_individuals)
        X = rng.normal(size=(ts.num_individuals, 4))

        sigma_grad_reml = nd.Derivative(lambda sigma: _explicit_reml(sigma, tau, ts, mu, y, X), n=1, step=1e-4)
        tau_grad_reml = nd.Derivative(lambda tau: _explicit_reml(sigma, tau, ts, mu, y, X), n=1, step=1e-4)
        ck_score = np.array([sigma_grad_reml(sigma), tau_grad_reml(tau)])
        np.testing.assert_allclose(ck_score, _explicit_gradient(sigma, tau, ts, mu, y, X))


class TestTSLMM:
    """
    just testing usage, correctness checks under `tsblup/validation`
    """

    def test_tslmm_usage(self):
        rng = np.random.default_rng(1)
        mu = 1e-10
        ts = msprime.sim_ancestry(100, sequence_length=1e5, recombination_rate=1e-8, population_size=1e4, random_seed=1024)
        subset = np.arange(ts.num_individuals)[::4]
        y = rng.normal(size=subset.size)
        X = rng.normal(size=(subset.size, 4))
        lmm = tslmm(ts, mu, phenotypes=y, covariates=X, phenotyped_individuals=subset, sgd_verbose=True, sgd_iterations=10)
        genetic_values = lmm.predict(np.arange(ts.num_individuals))
        assert len(lmm._optimization_trajectory) == 10
        assert genetic_values.size == ts.num_individuals

    def test_tslmm_usage_no_opt(self):
        rng = np.random.default_rng(1)
        sigma = 1.5
        tau = 0.5
        mu = 1e-10
        ts = msprime.sim_ancestry(100, sequence_length=1e5, recombination_rate=1e-8, population_size=1e4, random_seed=1024)
        subset = np.arange(ts.num_individuals)[::4]
        y = rng.normal(size=subset.size)
        X = rng.normal(size=(subset.size, 4))
        lmm = tslmm(ts, mu, phenotypes=y, covariates=X, phenotyped_individuals=subset, variance_components=np.array([sigma, tau]))
        genetic_values = lmm.predict(np.arange(ts.num_individuals))
        assert len(lmm._optimization_trajectory) == 0
        assert genetic_values.size == ts.num_individuals
