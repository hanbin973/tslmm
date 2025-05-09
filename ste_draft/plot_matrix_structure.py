
import numba
import msprime 
from linear_operators import TraitCovariance

num_threads = 4
numba.set_num_threads(num_threads)

n_samples = 50
n_pops = 20
s_length = 1e5
ne = 1e4

island_model = msprime.Demography.island_model([ne] * n_pops, 1e-5 / n_pops)
for i in range(1, n_pops):
    island_model.add_mass_migration(time=2*ne, source=i, dest=0, proportion=1.0)

ts = msprime.sim_ancestry(
    samples={f"pop_{i}": n_samples for i in range(n_pops)},
    recombination_rate=1e-8,
    sequence_length=s_length,
    demography=island_model,
    random_seed=1024,
)
ts.dump("example.ts")
covariance = TraitCovariance(ts, mutation_rate=1e-10)
mat = covariance.as_matrix(1.5, 0.5)

import matplotlib.pyplot as plt
img = plt.matshow(mat)
plt.colorbar(img)
plt.title(f"covariance matrix:\n{n_samples} samples per {n_pops} pops,\n{s_length/1e6:.2f} Mb")
plt.savefig("figs/covariance_matrix.png")
