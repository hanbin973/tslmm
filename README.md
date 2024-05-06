# tsblup 
**tsblup** is a complex trait analysis toolkit based on the `tskit` succinct tree sequence ecosystem.

It fits the following linear mixed model:

$$\mathbf{y} = X\mathbf{b} + Z\mathbf{u} + \boldsymbol{\varepsilon}$$

- $N$, $F$, and $E$ are the number of individuals, fixed effects, and edges (of the tree sequence).
- $\mathbf{y} \in \mathbb{R}^N$, $X \in \mathbb{R}^{N \times F}$, and $Z \in \mathbb{R}^{N \times E}$ are the trait vector, fixed effects design matrix, and random effects design matrix.
- $\mathbf{b} \in \mathbb{R}^F$ and $\mathbf{u} \in \mathbb{R}^E$ are fixed and random effects coefficients.
$\mathbf{u} \sim \mathcal{N}\left(\mathbf{0}, Q_{\mathbf{u}}^{-1}\right)$ where $Q_{\mathbf{u}}$ is the precision matrix.
- $\boldsymbol{\varepsilon}$ is the non-genetic error terms.

**tsblup** constructs $Z$ and $Q$ using the tree sequence.
Both are sparse matrices with $O(\text{number of edges})$ non-zero elements that can be stored conveniently.
See the example below:
```
import tskit
import msprime

import numpy as np
import pandas as pd
import scipy.io as io

import tsblup.operations as operations
import tsblup.matrices as matrices

# simulate tree sequence
ts = msprime.sim_ancestry(
    samples=1_000,
    recombination_rate=1e-8,
    sequence_length=100_000,
    population_size=10_000,
    random_seed=100
)

# break edges to have a unique subtree
break_ts = operations.split_upwards(ts)

# calculate Z (random effects design matrix) from tree sequence
Z = matrices.edge_individual_matrix(break_ts).T

# calculate Q (precision matrix) from tree sequence
A = matrices.edge_adjacency(break_ts).T
T = sparse.identity(break_ts.num_edges) - A
Q = T.T @ T

# export Z and Q 
io.mmwrite(Z, 'Z.mtx')
io.mmwrite(Q, 'Q.mtx')
```

`tsblup` can perform principal component analysis (PCA) on the tree sequence.
One can flexibly select branches by setting their weight.
A typical choice is to weight branches according to their area.

```
# edge removal & truncation
edges_weight = np.sqrt(edges_area)

# Design & RSVD
design = RowEdgeDesign(Z, T, edges_weight)
U, S, V = randomized_svd(design)
```

Note that zeros in the branch weights are not allowed. 
Instead, put a small number (e.g. `1e-10`).
