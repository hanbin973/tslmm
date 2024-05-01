import tskit
import msprime
import numpy as np
import scipy.sparse as sparse
import scipy
import sys
import scipy.io as io

import tsblup.operations as operations
import tsblup.matrices as matrices

# simulate tree sequence
ts = msprime.sim_ancestry(
    samples=100,
    recombination_rate=1e-8,
    sequence_length=5e4,
    population_size=1e4,
    random_seed=1024,
)

# break edges to have a unique subtree
break_ts = operations.split_upwards(ts)

# calculate Z (random effects design matrix) from tree sequence
Z = matrices.edge_individual_matrix(break_ts).T

# calculate Q (precision matrix) from tree sequence
A = matrices.edge_adjacency(break_ts).T
T = sparse.identity(break_ts.num_edges) - A
Q = T.T @ T

# export Z and T
io.mmwrite('Z.mtx', Z)
io.mmwrite('T.mtx', T)
