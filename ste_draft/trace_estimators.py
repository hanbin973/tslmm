from scipy.sparse.linalg import LinearOperator
from numpy.random import Generator
from typing import Callable
import numpy as np

# TODO use rng


def hutchinson(A: Callable, N: int, m: int, rng: Generator = None) -> [float, float]:
    """
    Hutchinson from https://arxiv.org/pdf/2301.07825
    """
    if rng is None: rng = np.random.default_rng()
    m = int(np.floor(m / 2))
    cnormc = lambda M: M / np.linalg.norm(M, 2, axis=0)[np.newaxis, :]  # divide by column norm
    diag_prod = lambda A, B: np.sum(A * B, axis=0)  # diag(A' @ B)

    # trace estimate
    Om = np.sqrt(N) * cnormc(rng.normal(size=(N, m)))
    Y = A(Om)
    ests = diag_prod(Om, Y)
    t = np.mean(ests)
    err = np.std(ests, ddof=1) / np.sqrt(m)

    return t, err


def xtrace(A: Callable, N: int, m: int, rng: Generator = None) -> [float, float]:
    """
    XTrace from https://arxiv.org/pdf/2301.07825
    """
    if rng is None: rng = np.random.default_rng()
    m = int(np.floor(m / 2))
    cnormc = lambda M: M / np.linalg.norm(M, 2, axis=0)[np.newaxis, :]  # divide by column norm
    diag_prod = lambda A, B: np.sum(A * B, axis=0)  # diag(A' @ B)

    # svd
    Om = np.sqrt(N) * cnormc(rng.normal(size=(N, m)))
    Y = A(Om)
    Q, R = np.linalg.qr(Y)

    # normalisation
    W = Q.T @ Om
    S = cnormc(np.linalg.inv(R).T)
    scale = (N - m + 1) / (N - np.linalg.norm(W, 2, axis=0) ** 2 \
        + np.abs(diag_prod(S, W) * np.linalg.norm(S, 2, axis=0)) ** 2)

    # trace estimate
    Z = A(Q)
    H = Q.T @ Z
    HW = H @ W
    T = Z.T @ Om
    dSW = diag_prod(S, W)
    dSHS = diag_prod(S, H @ S)
    dTW = diag_prod(T, W)
    dWHW = diag_prod(W, HW)
    dSRmHW = diag_prod(S, R - HW) 
    dTmHRS = diag_prod(T - H.T @ W, S)

    ests = np.sum(H.diagonal()) * np.ones(m) - dSHS + (dWHW - dTW + dTmHRS * dSW + \
        np.abs(dSW)**2 * dSHS + dSW.T * dSRmHW) * scale
    t = np.mean(ests)
    err = np.std(ests, ddof=1) / np.sqrt(m)

    return t, err


def xnystrace(A: Callable, N: int, m: int, rng: Generator = None) -> [float, float]:
    """
    XNysTrace from https://arxiv.org/pdf/2301.07825
    NB: `A` must be positive definite, this is not checked
    """
    if rng is None: rng = np.random.default_rng()
    cnormc = lambda M: M / np.linalg.norm(M, 2, axis=0)[np.newaxis, :]  # divide by column norm
    diag_prod = lambda A, B: np.sum(A * B, axis=0)  # diag(A' @ B)

    # nystrom
    Om = np.sqrt(N) * cnormc(np.random.randn(N, m))
    Y = A(Om)
    nu = np.finfo(float).eps / np.sqrt(N) * np.linalg.norm(Y)
    Y += nu * Om
    Q, R = np.linalg.qr(Y)
    H = Om.T @ Y
    C = np.linalg.cholesky((H + H.T) / 2).T
    B = np.linalg.solve(C.T, R.T).T

    # normalisation
    QQ, RR = np.linalg.qr(Om)
    WW = QQ.T @ Om
    SS = cnormc(np.linalg.inv(RR).T)
    scale = (N - m + 1) / (N - np.linalg.norm(WW, 2, axis=0) ** 2 \
        + np.abs(diag_prod(SS, WW).T * np.linalg.norm(SS, 2, axis=0)) ** 2)

    # trace estimate
    W = Q.T @ Om
    S = np.linalg.solve(C, B.T).T * (np.diag(np.linalg.inv(H)).T) ** (-1/2)
    dSW = diag_prod(S, W).T
    ests = np.linalg.norm(B) ** 2 - np.linalg.norm(S, 2, axis=0) ** 2 + \
        np.abs(dSW) ** 2 * scale - nu * N
    t = np.mean(ests)
    err = np.std(ests, ddof=1) / np.sqrt(m)

    return t, err


def xdiag(A: Callable, N: int, m: int, rng: Generator = None) -> np.ndarray:
    """
    XDiag from https://arxiv.org/pdf/2301.07825
    Assumes A is self-adjoint but this is not checked
    """
    if rng is None: rng = np.random.default_rng()
    m = int(np.floor(m / 2))
    cnormc = lambda M: M / np.linalg.norm(M, 2, axis=0)[np.newaxis, :]  # divide by column norm
    diag_prod = lambda A, B: np.sum(A * B, axis=0)  # diag(A' @ B)

    # randomized SVD
    Om = -3 + 2 * np.random.randint(1, 3, size=(N, m))  # Rademacher vectors
    Y = A(Om)
    Q, R = np.linalg.qr(Y)
    Z = A(Q)  # Z = A.T @ Q -- we're assuming A is self-adjoint
    T = Z.T @ Om
    S = cnormc(np.linalg.inv(R).T)
    dQZ = diag_prod(Q.T, Z.T)
    dQSSZ = diag_prod((Q @ S).T, (Z @ S).T)
    dOmQT = diag_prod(Om.T, (Q @ T).T)
    dOmY = diag_prod(Om.T, Y.T)
    dOmQSST = diag_prod(Om.T, (Q @ S @ np.diag(diag_prod(S, T))).T)

    # diagonal estimate
    d = dQZ + (-dQSSZ + dOmY - dOmQT + dOmQSST) / m

    return d
