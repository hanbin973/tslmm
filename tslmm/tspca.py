from __future__ import annotations

import numpy as np
import scipy.sparse

def _rand_pow_range_finder(
        operator: Callable,
        operator_dim: int,
        rank: int,
        depth: int,
        num_vectors: int,
        rng: np.random.Generator,
        ) -> np.ndarray:
    """
    Algorithm 9 in https://arxiv.org/pdf/2002.01387
    """
    assert num_vectors >= rank > 0
    test_vectors = rng.normal(size=(operator_dim, num_vectors))
    Q = test_vectors
    for i in range(depth):
        Q = np.linalg.qr(Q).Q
        Q = operator(Q)
    Q = np.linalg.qr(Q).Q
    return Q[:, :rank]

def _rand_svd(
        operator: Callable,
        operator_dim: int,
        rank: int,
        depth: int,
        num_vectors: int,
        rng: np.random.Generator,
        ) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Algorithm 8 in https://arxiv.org/pdf/2002.01387
    """
    assert num_vectors >= rank > 0
    Q = _rand_pow_range_finder(
            operator,
            operator_dim,
            num_vectors,
            depth,
            num_vectors,
            rng
            )
    C = operator(Q).T
    U_hat, D, V = np.linalg.svd(C, full_matrices=False)
    U = Q @ U_hat
    return U[:,:rank], D[:rank], V[:rank]
