# tslmm
**tslmm** is a complex trait analysis toolkit based on the `tskit` succinct tree sequence ecosystem.

It fits the following linear mixed model:

$$\mathbf{y} = X\mathbf{b} + Z\mathbf{u} + \boldsymbol{\varepsilon}$$

- $N$, $F$, and $E$ are the number of individuals, fixed effects, and edges (of the tree sequence).
- $\mathbf{y} \in \mathbb{R}^N$, $X \in \mathbb{R}^{N \times F}$, and $Z \in \mathbb{R}^{N \times E}$ are the trait vector, fixed effects design matrix, and random effects design matrix.
- $\mathbf{b} \in \mathbb{R}^F$ and $\mathbf{u} \in \mathbb{R}^E$ are fixed and random effects coefficients.
$\mathbf{u} \sim \mathcal{N}\left(\mathbf{0}, Q_{\mathbf{u}}^{-1}\right)$ where $Q_{\mathbf{u}}$ is the precision matrix.
- $\boldsymbol{\varepsilon}$ is the non-genetic error terms.

To use the program, first declare the `tslmm` object.
```
import numpy as np
from tslmm.tslmm import tslmm

mutation_rate = 1e-10
rng = np.random.default_rng()
# define tslmm object
# trait: 1d np.ndarray, covariates: 2d np.ndarray
lmm = tslmm(ts, mutation_rate, traits, covariates, rng)

# fit variance component
lmm.fit_variance_components(method='ai', haseman_elston=True, verbose=True)
```
