"""
Some gradient derivations for REML

deviance = logdet(G) + logdet(X' G^{-1} X) + (y - y_hat)' G^{-1} (y - y_hat)

==> trace(G^{-1} dG) - trace((X' G^{-1} X)^{-1} X' G^{-1} dG G^{-1} X)
    - (y - y_hat)' G^{-1} dG G^{-1} (y - y_hat) - 2 d_y_hat' G^{-1} (y - y_hat)
"""

import numpy as np
import numdifftools as nd

rng = np.random.default_rng(1)

print("-----------")

# b_hat = (X' G^{-1} X)^{-1} X' G^{-1} y
# d_b_hat = (X' G^{-1} X)^{-1} X' G^{-1} dG X X' G^{-1} y + (X' G^{-1} X)^{-1} X' G^{-1} dG y

n, m = 20, 3
y = rng.normal(size=(n, 1))
X = rng.normal(size=(n, m))
G = rng.normal(size=(n, 1000))
G = G @ G.T / 1000
s = 1.5
t = 0.5

Ga = G * t + np.eye(y.size) * s
GinvX = np.linalg.solve(Ga, X)
GinvY = np.linalg.solve(Ga, y)

def b_hat(X, G, y, s, t):
    G = G * t + np.eye(y.size) * s
    GinvX = np.linalg.solve(G, X)
    GinvY = np.linalg.solve(G, y)
    return np.linalg.solve(GinvX.T @ X, X.T @ GinvY)

def b_hat_grad(X, G, y, s, t):
    # y_hat = X (X' G^{-1} X)^{-1} X' G^{-1} y
    # d_b_hat = (X' G^{-1} X)^{-1} X' G^{-1} dG G^{-1} y_hat - (X' G^{-1} X)^{-1} X' G^{-1} dG G^{-1} y
    dGds = np.eye(y.size)
    dGdt = G
    G = G * t + np.eye(y.size) * s
    GinvX = np.linalg.solve(G, X) # n x m
    GinvY = np.linalg.solve(G, y) # n x 1
    b_hat = np.linalg.solve(GinvX.T @ X, X.T @ GinvY) # m x 1
    GinvYh = np.linalg.solve(G, X @ b_hat) # n x 1
    GinvXX = np.linalg.solve(GinvX.T @ X, GinvX.T) # m x n
    grad_s = GinvXX @ dGds @ (GinvYh - GinvY) # m x 1
    grad_t = GinvXX @ dGdt @ (GinvYh - GinvY) # m x 1
    return grad_s, grad_t


jac_s = nd.Jacobian(lambda s: b_hat(X, G, y, s, t), n=1, step=1e-4)
jac_t = nd.Jacobian(lambda t: b_hat(X, G, y, s, t), n=1, step=1e-4)
print(jac_s(s), jac_t(t), "\n", b_hat_grad(X, G, y, s, t))
print("-----------")

def REML(X, G, y, s, t):
    G = G * t + np.eye(y.size) * s
    GinvX = np.linalg.solve(G, X)
    GinvY = np.linalg.solve(G, y)
    b_hat = np.linalg.solve(GinvX.T @ X, X.T @ GinvY)
    y_hat = X @ b_hat
    resid = y - y_hat
    GinvR = np.linalg.solve(G, resid)
    sign, Gldet = np.linalg.slogdet(G)
    assert sign > 0
    sign, Xldet = np.linalg.slogdet(X.T @ GinvX)
    assert sign > 0
    return Gldet + Xldet + resid.T @ GinvR

def REML_grad(X, G, y, s, t):
    dGds = np.eye(y.size)
    dGdt = G
    G = G * t + np.eye(y.size) * s
    # do the below with qr(X)
    GinvX = np.linalg.solve(G, X) # n x m
    GinvY = np.linalg.solve(G, y) # n x 1
    b_hat = np.linalg.solve(GinvX.T @ X, X.T @ GinvY) # m x 1
    # GinvX.T @ X ==> GinvQ.T @ Q if we have QR
    GinvR = np.linalg.solve(G, y - X @ b_hat) # n x 1
    GinvXX = np.linalg.solve(GinvX.T @ X, GinvX.T) # m x n
    # X @ GinvXX ==> Q @ solve(GinvQ.T @ Q, GinvQ.T) 
    dy_ds = X @ (GinvXX @ (dGds @ GinvR)) # n x 1
    dy_dt = X @ (GinvXX @ (dGdt @ GinvR)) # n x 1
    # ---
    one = np.sum(np.linalg.solve(G, dGds).diagonal()) # use STE
    two = np.sum((GinvXX @ dGds @ GinvX).diagonal())  # exact
    thr = np.dot(GinvR.T, dGds @ GinvR) + 2 * dy_ds.T @ GinvR
    grad_s = one - two - thr
    one = np.sum(np.linalg.solve(G, dGdt).diagonal()) # use STE
    two = np.sum((GinvXX @ dGdt @ GinvX).diagonal()) # exact
    thr = np.dot(GinvR.T, dGdt @ GinvR) + 2 * dy_dt.T @ GinvR
    grad_t = one - two - thr
    # ---
    return grad_s, grad_t

jac_s = nd.Derivative(lambda s: REML(X, G, y, s, t), n=1, step=1e-4)
jac_t = nd.Derivative(lambda t: REML(X, G, y, s, t), n=1, step=1e-4)
print(jac_s(s), jac_t(t), "\n", REML_grad(X, G, y, s, t))

# this requires solving m + 1 more linear systems by CG
# we can do this in parallel, if m is not too big
