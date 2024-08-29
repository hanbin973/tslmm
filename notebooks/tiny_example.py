#!/usr/bin/env python
 
# ---- Imports ----

import io
import os
import sys
from IPython.display import SVG

import numpy as np
import numba
import scipy.sparse as sparse

import tskit

import tsblup.operations as operations
import tsblup.matrices as matrices

# ---- Generate a tree sequence example ----

nodes = io.StringIO(
    """\
    id is_sample time individual
     0         1    0          0
     1         1    0          0
     2         1    0          1
     3         1    0          1
     4         0    1         -1
     5         0    1         -1
     6         0    2         -1
     7         0    4         -1
    """
)

edges = io.StringIO(
    """\
    left right parent child
       0    10      4     0
       0     5      4     1
       5    10      5     1
       0    10      5     2
       0    10      6     3
       0    10      6     5
       0    10      7     4
       0    10      7     6
    """
)

individuals = io.StringIO(
    """\
    flags location parents
        0        0      -1
        0        0      -1
    """
)

ts = tskit.load_text(nodes = nodes, edges = edges, individuals = individuals, strict = False)
print(ts)
SVG(ts.draw_svg())

# ---- Split-upwards the tree sequence ----

ts.num_edges # before the split
ts.dump_tables().edges

ts = operations.split_upwards(ts)

ts.num_edges # after the split
ts.dump_tables().edges
print(ts)
SVG(ts.draw_svg())

# ---- Setup matrices ----

# The assumed model here is:
# 
# y = Xb + Zg + e
# 
# y (n_y * 1) phenotypic values
# X (n_y * n_b) design matrix linking y with b
# b (n_b * 1) fixed effects
# Z (n_y * n_g) design matrix linking y with g
# g (n_g * 1) individuals' genetic values
# e (n_y * 1) residuals; e ~ N(0, I sigma^2_e)
# 
# g = Wr
# W (n_g * n_edges) design matrix linking g with r
# r (n_edges * 1) edge values
# 
# r = Tu
# T (n_edges * n_edges) design matrix linking r with u (summing up u into edge values)
# u (n_edges * 1) edge "innovation" effects on edge areas; u ~ N(0, D sigma^2_u)
# (due to new mutations arising along the DNA span of edges between "parent" and "child"
#  nodes and along the time length of edges --> edge areas)
# 
# r = TSu*; u* ~ N(0, I sigma^2_u)
# T (n_edges * n_edges) edge design matrix linking r with u* (summing up u* into edge values)
# S (n_edges * n_edges) diagonal matrix with sqrt(edge areas)
# u (n_edges * 1) normalised edge "innovation" effects; u* ~ N(0, sigma^2_u)
# 
# Covariance between genetic values:
# 
# C_g = Var(g | W, T, S) = Var(Wr | W, T, S) = Var(WTSu* | W, T, S)
#                        = WTS Var(u*) S'T'W' = WTDT'W' sigma^2_u = G sigma^2_u
# 
# Covariance between edge values:
# 
# C_r = Var(r | T, S) = TDT' sigma^2_u = R sigma^2_u
#
# TODO: r and R might be bad letters? (r could be tought of as a residual?)
# 
# Some relations:
#  
# Q_r = inv(C_r) = inv(TDT' sigma^2_u) = inv(LL' sigma^2_u)
#     = inv(T)' inv(D) inv(T) sigma^(-2)_u = inv(L)' inv(L) sigma^(-2)_u
#     = (I - E)' inv(D) (I - E) sigma^(-2)_u
#     = (I - E)' inv(SS) (I - E) sigma^(-2)_u
# inv(L) = inv(S) (I - E) = inv(S) inv(T)
# L = inv(inv(L)) = inv(inv(S) inv(T)) = TS
# inv(T) = I - E

# Design matrix linking design matrix linking y with b
X = np.array([1, 1])

# Design matrix linking individuals and edges (those that point to individuals' nodes)
# (at the moment, W matrix here combines Z & W matrices above)
W = matrices.edge_individual_matrix(ts).T
W.shape
print(W)

# Edge value "structure" matrix (TInv = I-E) (adjacent edges)
TInv = sparse.identity(ts.num_edges) - matrices.edge_adjacency(ts).T
TInv.sort_indices() # ensure order for working across Python packages
TInv.shape
print(TInv.toarray()) # it's upper-triangular because we order/number edges back in time

# Edge value design matrix (T = inv(I-E)) (edge connections back in time)
T = np.linalg.inv(TInv.toarray())
print(T) # it's upper-triangular because we order/number edges back in time

# Diagonal matrix of variance or precision coefficients of edge effects
edge_spans = ts.edges_right - ts.edges_left
edge_lengths = ts.nodes_time[ts.edges_parent] - ts.nodes_time[ts.edges_child]
edge_areas = edge_spans * edge_lengths # = mutational_target
D = sparse.diags_array(edge_areas)
S = np.sqrt(D)
DInv = sparse.diags_array(1 / edge_areas)
SInv = np.sqrt(DInv)
print(np.column_stack((edge_spans, edge_lengths, edge_areas, (1 / edge_areas).round(3))))

# Precision coefficient matrix for edge effects per unit area (note the missing * sigma^(-2)_u)
Qus = sparse.identity(ts.num_edges)
print(Qus.toarray().round(3))
# See below for print((Qus.toarray() / sigma2u).round(3))

# Covariance coefficient matrix for edge effects per unit area (note the missing * sigma^2_u)
Cus = sparse.identity(ts.num_edges).toarray()
print(Cus.round(3))
# See below for print((Qu / sigma2u).round(3))

# Precision coefficient matrix for edge effects (note the missing * sigma^(-2)_u)
Qu = DInv
print(Qu.toarray().round(3))
# See below for print((Qu.toarray() / sigma2u).round(3))

# Covariance coefficient matrix for edge effects (note the missing * sigma^2_u)
Cu = np.linalg.inv(Qu.toarray())
print(Cu.round(3))
# See below for print((Qu / sigma2u).round(3))

# Cholesky factor of precision coefficient matrix for edge values
LInv = SInv @ TInv
print(LInv.toarray().round(3))

# Cholesky factor of covariance coefficient matrix for edge values
L = np.linalg.inv(LInv.toarray())
print(L.round(3))
# print((T @ S).round(3))

# Precision coefficient matrix for edge values (note the missing * sigma^(-2)_u)
Qr = LInv.T @ LInv
print(Qr.toarray().round(3))
# See below for print((Qr.toarray() / sigma2u).round(3))

# Covariance coefficient matrix for edge values (note the missing * sigma^2_u)
Cr = np.linalg.inv(Qr.toarray())
print(Cr.round(3))
Cr = L @ L.T
print(Cr.round(3))
# See below for print((Cr / sigma2u).round(3))

# Covariance coefficient matrix for node values (note the missing * sigma^2_u)
N = matrices.edge_child_matrix(ts).T
print(N)
Cn = N @ Cr @ N.T
print(Cn.round(3))
# See below for print((Cn * sigma2u).round(3))
# The last row and column are full of 0s because the last node is the root

# Precision coefficient matrix for node values (note the missing * sigma^(-2)_u)
Qn = np.linalg.inv(Cn[0:(ts.num_nodes - 1), 0:(ts.num_nodes - 1)])
print(Qn.round(3))
# See below for print((Qn / sigma2u).round(3))

# Covariance coefficient matrix for genetic values (note the missing * sigma^2_u)
Cg = W @ Cr @ W.T
print(Cg.round(3))
# See below for print((Cg * sigma2u).round(3))

# Precision coefficient matrix for genetic values (note the missing * sigma^(-2)_u)
Qg = np.linalg.inv(Cg)
print(Qg.round(3))
# See below for print((Qg / sigma2u).round(3))

# ---- Simulate values ----

sigma2u = 0.001
print((Cus * sigma2u).round(3))
print((Qus.toarray() / sigma2u).round(3))

print((Cu * sigma2u).round(3))
print((Qu.toarray() / sigma2u).round(3))

print((Cr * sigma2u).round(3))
print((Qr.toarray() / sigma2u).round(3))

print((Cn * sigma2u).round(3))
print((Qn / sigma2u).round(3))

print((Cg * sigma2u).round(3))
print((Qg / sigma2u).round(3))

edge_effects = np.random.normal(size = ts.num_edges,
                                scale = np.sqrt(edge_areas * sigma2u)).round(3)
# For this pedagogical demonstration we use large & simple values for edge effects
# (you can swap with the above sampled edge effects instead)
#                        0,  1, 2, 3, 4, 5,  6, 7,  8, 9, 10
edge_effects = np.array([0, -1, 1, 0, 2, 0, -1, 2, -1, 0, -1])
print(edge_effects) # u 

edge_effects_per_unit_area = edge_effects / np.sqrt(edge_areas)
print(edge_effects_per_unit_area) # u*

edge_values = T @ edge_effects # r = Tu
print(edge_values)

node_values = N @ edge_values # n = Nr = NTu
print(node_values)

genetic_values = W @ edge_values # g = Wr = WTu (via edge values - canonical approach)
print(genetic_values)

M = matrices.node_individual_matrix(ts).T
print(M)
genetic_values2 = M @ node_values # g = Mn = MNr = MNTu (via node values)
print(np.column_stack((genetic_values, genetic_values2)))

sigma2e = 1
residuals = np.random.normal(size = ts.num_individuals,
                             scale = np.sqrt(sigma2e)).round()
# For this pedagogical demonstration we use small & simple values for residuals
# (you can swap with the above sampled residuals instead)
#                       0,    1
residuals = np.array([0.1, -0.1])
print(residuals)

intercept = 10
phenotype_values = intercept + genetic_values + residuals
print(phenotype_values)

# ---- Estimate effects and values (using simple pedagogic approach) ----

# Throughout we assume we know variance components!

# Phenotypic variance and precision
Cy = Cg * sigma2u + np.eye(N = len(phenotype_values)) * sigma2e
print(Cy)
Qy = np.linalg.inv(Cy)
print(Qy)

# Best linear estimate of intercept
intercept_hat = (X.T @ phenotype_values) / (X.T @ X)
print(np.column_stack((intercept, intercept_hat)))

# Best linear prediction of genetic values (g)
# cov(g, (y = g + e)') inv(var(y)) (y - E(y))
tmp = (np.linalg.inv(Cy) @ (phenotype_values - intercept_hat))
# cov(g, g') ...
genetic_values_hat = (Cg * sigma2u) @ tmp
print(np.column_stack((genetic_values, genetic_values_hat)))

# Best linear prediction of node values (n; g = Mn = MNr)
# cov(n, (y = g + e)') inv(var(y)) (y - E(y))
# cov(n, g' = nM') ...
node_values_hat = (Cn @ M.T * sigma2u) @ tmp
print(np.column_stack((node_values, node_values_hat)))

# Best linear prediction of edge values (r; g = Wr)
# cov(r, (y = g + e)') inv(var(y)) (y - E(y))
# cov(r, g' = rW') ...
edge_values_hat = (Cr @ W.T * sigma2u) @ tmp
print(np.column_stack((edge_values, edge_values_hat)))

# Best linear prediction of edge effects (u; g = WTu)
# cov(u, (y = g + e)') inv(var(y)) (y - E(y))
# cov(u, g' = uT'W') ...
edge_effects_hat = (Cu @ T.T @ W.T * sigma2u) @ tmp
print(np.column_stack((edge_effects, edge_effects_hat)))

# Best linear prediction of edge effects per unit area (u*; u = Su*)
edge_effects_per_unit_area_hat = edge_effects_hat / np.sqrt(edge_areas)
print(np.column_stack((edge_effects_per_unit_area, edge_effects_per_unit_area_hat)))

# System of equations for the linear mixed model
# TODO: why do we have differences in estimates? numerical or code errors?

XX = np.array(X.T @ X, ndmin = 2)
XW = np.array(X.T @ W, ndmin = 2)
WX = XW.T
WW = (W.T @ W).toarray() + Qr * sigma2e / sigma2u
LHS = np.concatenate((np.concatenate((XX, XW), axis = 1),
                      np.concatenate((WX, WW), axis = 1)),
                     axis = 0)

Xy = np.array(X.T @ phenotype_values, ndmin = 2)
Wy = np.array(W.T @ phenotype_values, ndmin = 2).T
RHS = np.concatenate((Xy, Wy), axis = 0)

solutions_hat = np.linalg.solve(a = LHS, b = RHS)

intercept_hat2 = solutions_hat[0]
print(np.column_stack((intercept, intercept_hat, intercept_hat2)))

edge_values_hat2 = solutions_hat[1:12]
print(np.column_stack((edge_values, edge_values_hat, edge_values_hat2)))

edge_effects_hat2 = TInv @ edge_values_hat2 # r = Tu --> inv(T) r = u
print(np.column_stack((edge_effects, edge_effects_hat, edge_effects_hat2)))

edge_effects_per_unit_area_hat2 = LInv @ edge_values_hat2 # r = TSu* = Lu* --> inv(L) r = u*
print(np.column_stack((edge_effects_per_unit_area, edge_effects_per_unit_area_hat, edge_effects_per_unit_area_hat2)))

node_values_hat2 = N @ edge_values_hat2 # n = Nr
print(np.column_stack((node_values, node_values_hat, node_values_hat2)))

genetic_values_hat2 = W @ edge_values_hat2 # g = Wr
print(np.column_stack((genetic_values, genetic_values_hat, genetic_values_hat2)))

# ---- Estimate effects and values (using tslmm) ----

y = phenotype_values - intercept_hat

sys.path.append(os.getcwd() + "/ste_draft")
from linear_operators import TraitCovariance, NystromPreconditioner
from genetic_values import edge_effects as edge_effects_fun

covmat = TraitCovariance(ts)
precon = NystromPreconditioner(covmat, rank = 1, samples = 10)
edge_effects_per_unit_area_cg_hat, _ = edge_effects_fun(sigma2e, sigma2u, y, covmat, None)
Xn = covmat.Z.toarray() @ np.linalg.inv(covmat.L.T.toarray())
post_var = np.linalg.inv(np.eye(covmat.factor_dim) / sigma2u + Xn.T @ Xn / sigma2e)
edge_effects_per_unit_area_exact_hat = post_var @ Xn.T @ y / sigma2e

print(np.column_stack((edge_effects_per_unit_area, edge_effects_per_unit_area_hat,
                       edge_effects_per_unit_area_exact_hat, edge_effects_per_unit_area_cg_hat)))
