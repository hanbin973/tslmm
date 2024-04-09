# tsblup 
**tstrait** is a complex trait analysis toolkit based on the `tskit` succinct tree sequence ecosystem.

It fits the following linear mixed model:

$$\mathbf{y} = X\mathbf{b} + Z\mathbf{u} + \boldsymbol{\varepsilon}$$

- $\mathbf{y} \in \mathbb{R}^N$, $X \in \mathbb{R}^{N \times F}$, and $Z \in \mathbb{R}^{N \times E}$ are the trait vector, fixed effects design matrix, and random effects design matrix.
- $b \in \mathbb{R}^F$ and $u \in \mathbb{R}^E$ are fixed and random effects coefficients.
- $\boldsymbol{\varepsilon}$ is the non-genetic error terms.
- $N$, $F$, and $E$ are the number of individuals, fixed effects, and edges (of the tree sequence).


