# tsblup 
*tstrait* is a complex trait analysis toolkit based on the `tskit` succinct tree sequence ecosystem.

It fits the following linear mixed model:
$$ \mathbf{y} = X\mathbf{b} + Z\mathbf{u} + \boldsymbol{\varepsilon} $$
- $y$, $X$ and $Z$ are the trait vector, fixed effects design matrix, and random effects design matrix.
- $b$ and $u$ are fixed and random effects coefficients.
- $\boldsymbol{\varepsilon}$ is the non-genetic error terms.


