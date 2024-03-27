import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba

import scipy.sparse as sparse
import scipy.io as io

import tskit as ts
import tstrait
import msprime
import ldgm

from ts_blup import _make_adjacency

###### simulate tree sequence
ts = msprime.sim_ancestry(
    samples=2000,
    recombination_rate=1e-8,
    sequence_length=100_000,
    population_size=10_000,
    random_seed=100,
)

# chunck edges to have unique descendant nodes
bts = ldgm.brick_ts(ts)

# simulate mutations
bts = msprime.sim_mutations(bts, rate=1e-7, random_seed=101)

# simulate traits
model = tstrait.trait_model(distribution="normal", mean=0, var=1)
sim_result = tstrait.sim_phenotype(
    ts=bts, num_causal=100, model=model, h2=0.5, random_seed=1
)

###### inverse of GRM (precision matrix Q)
A = _make_adjacency(bts) # A is upper triangular
T = sparse.identity(bts.num_edges) - A 

edges_area = (bts.edges_right - bts.edges_left) * (bts.nodes_time[bts.edges_parent] - bts.nodes_time[bts.edges_child])
edges_area = edges_area / np.mean(edges_area) # normalize edge area for INLA stability (workaround)
Dinv = sparse.diags(edges_area ** (-1))

Q = T.T @ Dinv @ T 

###### Random effects design matrix (design matrix Z)
edge_to_node = sparse.csr_matrix(
    (
        np.ones(bts.num_edges),
        bts.edges_child,
        np.arange(bts.num_edges+1)
    ),
    shape=(bts.num_edges, bts.num_nodes)
) # edge-node adjacency matrix
node_to_ind = sparse.csr_matrix(
    (
        np.ones(bts.num_samples,dtype=np.int32),
        bts.nodes_individual[:bts.num_samples],
        np.hstack((np.arange(bts.num_samples), np.full(bts.num_nodes-bts.num_samples+1,bts.num_samples)))
    ),
    shape=(bts.num_nodes, bts.num_individuals)
) # node - individual adjancency matrix

Z = (edge_to_node @ node_to_ind).T


###### save outputs
io.mmwrite('Ginv.mtx', Q)
io.mmwrite('Z.mtx', Z)
np.savetxt('pheno.txt', sim_result.phenotype.phenotype.values)



