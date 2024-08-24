import stdpopsim
import tskit

engine = stdpopsim.get_engine("slim")

species = stdpopsim.get_species("CanFam")
contig = species.get_contig('1', genetic_map="Campbell2016_CanFam3_1")

popsize_changes = [
        # (time-ago, size)
        (k, 2**(10-k-1) * 1000) for k in range(10)
] + [(100, 100), ]
model = stdpopsim.PiecewiseConstantSize(species.population_size, *popsize_changes)

samples = {"pop_0": 100000}

ts = engine.simulate(model, contig, samples, slim_burn_in=1000)

ts.dump("one_pop.trees")
