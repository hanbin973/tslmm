# tslmm
**tslmm** is a complex trait analysis toolkit based on the `tskit` succinct tree sequence ecosystem.

It fits the following linear mixed model:

$$\mathbf{y} = X\mathbf{b} + Z\mathbf{u} + \boldsymbol{\varepsilon}$$

- $N$, $F$, and $E$ are the number of individuals, fixed effects, and edges (of the tree sequence).
- $\mathbf{y} \in \mathbb{R}^N$, $X \in \mathbb{R}^{N \times F}$, and $Z \in \mathbb{R}^{N \times E}$ are the trait vector, fixed effects design matrix, and random effects design matrix.
- $\mathbf{b} \in \mathbb{R}^F$ and $\mathbf{u} \in \mathbb{R}^E$ are fixed and random effects coefficients.
$\mathbf{u} \sim \mathcal{N}\left(\mathbf{0}, Q_{\mathbf{u}}^{-1}\right)$ where $Q_{\mathbf{u}}$ is the precision matrix.
- $\boldsymbol{\varepsilon}$ is the non-genetic error terms.

See this [example](https://github.com/hanbin973/tslmm/blob/main/notebooks/prediction_example.ipynb) for full use.
