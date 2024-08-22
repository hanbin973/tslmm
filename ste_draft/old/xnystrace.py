import numpy as np
import scipy
import matplotlib.pyplot as plt

grid = np.arange(2, 200)
foo = np.abs(np.random.randn(1000, 1000))

#def hutchinson(A: LinearOperator, m: int) -> [float, float]:
def hutchinson(A, m):
    N = A.shape[0]
    m = int(np.floor(m / 2))
    cnormc = lambda M: M / np.linalg.norm(M, 2, axis=0)[np.newaxis, :]  # divide by column norm
    diag_prod = lambda A, B: np.sum(A * B, axis=0)  # diag(A' @ B)
    Om = np.sqrt(N) * cnormc(np.random.randn(N, m))
    Y = A @ Om 
    ests = diag_prod(Om, Y)
    t = np.mean(ests)
    err = np.std(ests, ddof=1) / np.sqrt(m)

    return t, err


uno = np.sum(foo.diagonal())
strm = [hutchinson(foo, i) for i in grid]
est = np.array([s for s, e in strm])
err = np.array([e for s, e in strm])
print("Last est", uno, est[-1])
print("95Cov", np.sum(np.logical_and(uno > est - 1.96 * err, uno < est + 1.96 * err)) / grid.size)

plt.scatter(grid, est, s=4, c="red")
plt.vlines(grid, est - 1.96 * err, est + 1.96 * err, color="red")
plt.axhline(y=uno, linestyle="--", color="black")
plt.savefig("hutchinson_convergence.png")
plt.clf()



#def xtrace(A: LinearOperator, m: int) -> [float, float]:
def xtrace(A, m):
    N = A.shape[0]
    m = int(np.floor(m / 2))
    cnormc = lambda M: M / np.linalg.norm(M, 2, axis=0)[np.newaxis, :]  # divide by column norm
    diag_prod = lambda A, B: np.sum(A * B, axis=0)  # diag(A' @ B)

    # svd
    Om = np.sqrt(N) * cnormc(np.random.randn(N, m))
    Y = A @ Om 
    Q, R = np.linalg.qr(Y)

    # normalisation
    W = Q.T @ Om
    S = cnormc(np.linalg.inv(R).T)
    scale = (N - m + 1) / (N - np.linalg.norm(W, 2, axis=0) ** 2 \
        + np.abs(diag_prod(S, W) * np.linalg.norm(S, 2, axis=0)) ** 2)

    # Trace estimate
    Z = A @ Q
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


uno = np.sum(foo.diagonal())
strm = [xtrace(foo, i) for i in grid]
est = np.array([s for s, e in strm])
err = np.array([e for s, e in strm])
print("Last est", uno, est[-1])
print("95Cov", np.sum(np.logical_and(uno > est - 1.96 * err, uno < est + 1.96 * err)) / grid.size)

plt.scatter(grid, est, s=4, c="red")
plt.vlines(grid, est - 1.96 * err, est + 1.96 * err, color="red")
plt.axhline(y=uno, linestyle="--", color="black")
plt.savefig("xtrace_convergence.png")
plt.clf()


# ---

foo = foo @ foo.T / 2 + foo.T @ foo / 2

#def xnystrace(A: LinearOperator, m: int) -> [float, float]:
def xnystrace(A, m):
    N = A.shape[0]
    cnormc = lambda M: M / np.linalg.norm(M, 2, axis=0)[np.newaxis, :]  # divide by column norm
    diag_prod = lambda A, B: np.sum(A * B, axis=0)  # diag(A' @ B)

    # nystrom
    Om = np.sqrt(N) * cnormc(np.random.randn(N, m))
    Y = A @ Om 
    nu = 1 / np.sqrt(N) * np.finfo(float).eps * np.linalg.norm(Y)
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

    # Trace estimate
    W = Q.T @ Om
    S = np.linalg.solve(C, B.T).T * (np.diag(np.linalg.inv(H)).T) ** (-1/2)
    dSW = diag_prod(S, W).T
    ests = np.linalg.norm(B) ** 2 - np.linalg.norm(S, 2, axis=0) ** 2 + np.abs(dSW) ** 2 * scale \
        - nu * N
    t = np.mean(ests)
    err = np.std(ests, ddof=1) / np.sqrt(m)

    return t, err




uno = np.sum(foo.diagonal())
strm = [xnystrace(foo, i) for i in grid]
est = np.array([s for s, e in strm])
err = np.array([e for s, e in strm])
print("Last est", uno, est[-1])
print("95Cov", np.sum(np.logical_and(uno > est - 1.96 * err, uno < est + 1.96 * err)) / grid.size)

plt.scatter(grid, est, s=4, c="red")
plt.vlines(grid, est - 1.96 * err, est + 1.96 * err, color="red")
plt.axhline(y=uno, linestyle="--", color="black")
plt.savefig("xnystrace_convergence.png")
plt.clf()


# ---

#def xdiag(A: LinearOperator, m: int) -> np.ndarray:
def xdiag(A, m):
    N = A.shape[0]
    m = int(np.floor(m / 2))
    cnormc = lambda M: M / np.linalg.norm(M, 2, axis=0)[None, :]  # divide by column norm
    diag_prod = lambda A, B: np.sum(A * B, axis=0)  # diag(A' @ B)

    # Randomized SVD
    Om = -3 + 2 * np.random.randint(1, 3, size=(N, m))  # Rademacher vectors
    Y = A @ Om
    Q, R = np.linalg.qr(Y)
    Z = A.T @ Q
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



uno = foo.diagonal()[0]
strm = [xdiag(foo, i)[0] for i in grid]
print(uno, strm[-1])

plt.scatter(grid, strm / uno)
plt.axhline(y=1.0, linestyle="--", color="black")
plt.savefig("xdiag_convergence.png")
plt.clf()

