import stdpopsim
import tskit
import numpy as np

L = 1e8

engine = stdpopsim.get_engine("msprime")

species = stdpopsim.get_species("CanFam")
contig = species.get_contig('1', genetic_map="Campbell2016_CanFam3_1", left=1e7, right=1e7 + L)

T = 100
N12a = 100
N12 = 50_000
M = 0.01
model = stdpopsim.IsolationWithMigration(
        NA=species.population_size, 
        N1=N12,
        N2=N12,
        T=T,
        M12=M,
        M21=M,
)
model.model.populations[0].growth_rate = -np.log(N12a / N12) / T
model.model.populations[1].growth_rate = -np.log(N12a / N12) / T

samples = { "pop1": 50_000, "pop2": 50_000 }

ts = engine.simulate(model, contig, samples)

ts.dump("two_pop.trees")

