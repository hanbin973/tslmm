import subprocess
import tskit
import tstrait
import msprime
import pickle
import tszip
import numpy as np
import os
import argparse
import warnings

import matplotlib.pyplot as plt


docstring = \
r"""
- Simulate ARG under spatial model with local density regulation using SLiM
- Write out pedigree information from tracked ancestral individuals into msprime.PedigreeBuilder
- Simulate conditional on pedigree with msprime
- Simulate neutral mutations on top of genealogies
- Assign effect size to mutations with tstrait
- Make some sanity checks (e.g. fine-scale spatial clustering of close relatives, phenotypes)
"""

slim_model = r"""
initialize() {
    initializeSLiMModelType("nonWF");
    initializeSLiMOptions(dimensionality="xy");
    
    // This model uses tree-sequence recording, but it is optional
    initializeTreeSeq();
    
    defaults = Dictionary(
        "SEED", 5,         // random seed
        "SD", 0.3,         // sigma_D, dispersal distance 
        "SX", 0.3,         // sigma_X, interaction distance for measuring local density
        "SM", 0.3,         // sigma_M, mate choice distance
        "K", 20,           // carrying capacity per unit area 20 10 5 5 5
        "LIFETIME", 4,     // average life span
        "WIDTH", 25.0,     // width of the simulated area
        "HEIGHT", 25.0,    // height of the simulated area
        "RUNTIME", 200000, // total number of ticks to run the simulation for
        "L", 1e8,          // genome length
        "R", 1e-8,         // recombination rate
        "MU", 0            // mutation rate
    );  
    
    // Set up parameters with a user-defined function
    setupParams(defaults);
    
    // Set up constants that depend on externally defined parameters
    defineConstant("FECUN", 1 / LIFETIME);
    defineConstant("RHO", FECUN / ((1 + FECUN) * K));
    defineConstant("PARAMS", defaults);
    
    setSeed(SEED);
    
    // basic neutral genetics
    initializeMutationRate(MU);
    initializeMutationType("m1", 0.5, "f", 0.0);
    initializeGenomicElementType("g1", m1, 1.0);
    initializeGenomicElement(g1, 0, L-1);
    initializeRecombinationRate(R);
    
    // spatial interaction for local density measurement
    initializeInteractionType(1, "xy", reciprocal=T, maxDistance=3 * SX);
    i1.setInteractionFunction("n", 1, SX);
    
    // spatial interaction for mate choice
    initializeInteractionType(2, "xy", reciprocal=T, maxDistance=3 * SM);
    i2.setInteractionFunction("n", 1, SM);
}

1 first() {
    sim.addSubpop("p0", asInteger(K * WIDTH * HEIGHT));
    p0.setSpatialBounds(c(0, 0, WIDTH, HEIGHT));
    p0.individuals.setSpatialPosition(p0.pointUniform(p0.individualCount));

    // Remember founders
    sim.treeSeqRememberIndividuals(p0.individuals, permanent=T);
}

first() {
    // preparation for the reproduction() callback
    i2.evaluate(p0);
}

reproduction() {
    mate = i2.drawByStrength(individual, 1);
    if (mate.size())
        subpop.addCrossed(individual, mate, count=rpois(1, FECUN));
}

early() {
    // Disperse offspring
    offspring = p0.subsetIndividuals(maxAge=0);
    p0.deviatePositions(offspring, "reprising", INF, "n", SD);
        
    // Measure local density and use it for density regulation
    i1.evaluate(p0);
    inds = p0.individuals;
    competition = i1.localPopulationDensity(inds);
    inds.fitnessScaling = 1 / (1 + RHO * competition);
}

late() {
    if (p0.individualCount == 0) {
        catn("Population went extinct! Ending the simulation.");
        sim.simulationFinished();
    }

    // Remember individuals
    sim.treeSeqRememberIndividuals(p0.individuals, permanent=T);
}

RUNTIME late() {
    catn("End of simulation (run time reached)");
    sim.treeSeqOutput(OUTPATH, metadata=PARAMS);
    sim.simulationFinished();
}

function (void)setupParams(object<Dictionary>$ defaults)
{
    if (!exists("PARAMFILE")) defineConstant("PARAMFILE", "./params.json");
    if (!exists("OUTDIR")) defineConstant("OUTDIR", ".");
    defaults.addKeysAndValuesFrom(Dictionary("PARAMFILE", PARAMFILE, "OUTDIR", OUTDIR));
    
    if (fileExists(PARAMFILE)) {
        defaults.addKeysAndValuesFrom(Dictionary(readFile(PARAMFILE)));
        defaults.setValue("READ_FROM_PARAMFILE", PARAMFILE);
    }
    
    defaults.setValue("OUTBASE", OUTDIR + "/out_" + defaults.getValue("SEED"));
    defaults.setValue("OUTPATH", defaults.getValue("OUTBASE") + ".trees");
    
    for (k in defaults.allKeys) {
        if (!exists(k))
            defineConstant(k, defaults.getValue(k));
        else
            defaults.setValue(k, executeLambda(k + ";"));
    }
    
    // print out default values
    catn("===========================");
    catn("Model constants: " + defaults.serialize("pretty"));
    catn("===========================");
}
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(docstring)
    parser.add_argument(
        "--seed", type=int, default=1024,
        help="Random seed",
    )
    parser.add_argument(
        "--spatial-kernel", type=float, default=0.25,
        help="StdDev of dispersal/interaction/mate choice distance "
        "if this is too small, population will go extinct",
    )
    parser.add_argument(
        "--carrying-cap", type=int, default=5,
        help="Local carrying capacity, "
        "if this is too small, population will go extinct",
    )
    parser.add_argument(
        "--spatial-extent", type=float, default=100.0, #default=500.0,
        help="Width of the square spatial domain",
    )
    parser.add_argument(
        "--sequence-length", type=float, default=1e8,
        help="Sequence length in bp",
    )
    parser.add_argument(
        "--recombination-rate", type=float, default=1e-8,
        help="Recombination rate",
    )
    parser.add_argument(
        "--mutation-rate", type=float, default=1.0e-9,
        help="Mutation rate",
    )
    parser.add_argument(
        "--num-causal-mutations", type=int, default=10000,
        help="Number of causal mutations",
    )
    parser.add_argument(
        "--heritability", type=float, default=1.0,
        help="Narrow sense heritability",
    )
    parser.add_argument(
        "--ticks", type=int, default=200,
        help="Number of ticks to run pedigree simulation",
    )
    parser.add_argument(
        "--out-prefix", type=str, default="spatial-simulation",
        help="Output path for tree sequence",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite all output",
    )
    parser.add_argument(
        "--overwrite-from-msp", action="store_true",
        help="Overwrite output from msprime simulation onwards",
    )
    args = parser.parse_args()
    logfile = open(args.out_prefix + ".log", "w")

    # SLiM simulation of the (spatial) pedigree
    slim_pedigree_path = args.out_prefix + ".pedigree.trees"
    slim_args = [
        "slim",
        "-d", f"SEED={args.seed}",
        "-d", f"SD={args.spatial_kernel}",
        "-d", f"SX={args.spatial_kernel}",
        "-d", f"SM={args.spatial_kernel}",
        "-d", f"K={args.carrying_cap}",
        "-d", f"LIFETIME=1",
        "-d", f"WIDTH={args.spatial_extent}",
        "-d", f"HEIGHT={args.spatial_extent}",
        "-d", f"RUNTIME={args.ticks}",
        "-d", f"L=1",
        "-d", f"R=0.0",
        "-d", f"OUTPATH='{slim_pedigree_path}'",
    ]
    if not os.path.exists(slim_pedigree_path) or args.overwrite:
        if os.path.exists(slim_pedigree_path): os.remove(slim_pedigree_path)
        result = subprocess.run(slim_args, input=slim_model.encode("ascii"), capture_output=True)
        logfile.write(result.stdout.decode("utf-8"))
        if result.returncode:
            raise ValueError(result.stderr)
        tszip.compress(tskit.load(slim_pedigree_path), slim_pedigree_path)
    

    pedigree_tables_path = args.out_prefix + ".pedigree.tables"
    if not os.path.exists(pedigree_tables_path) or args.overwrite:
        # all individuals are recorded in this tree sequence, whether or not these contribute
        # ancestry to the contemporary individuals, and all of these are marked as samples
        #slim_ts = tskit.load(slim_pedigree_path)
        slim_ts = tszip.decompress(slim_pedigree_path)

        # we'll only record pedigree of newborns
        sample_codes = [
            ind.metadata["pedigree_id"] for ind in slim_ts.individuals()
            if slim_ts.nodes_time[ind.nodes[0]] == 0
        ]
        logfile.write(f"Pedigree has {len(sample_codes)} contemporary samples\n")

        # dump the full pedigree in terms of SLiM codes
        parents = {}
        ages = {}
        locations = {}
        sex = {}
        for ind in slim_ts.individuals():
            ch, p1, p2 = ind.metadata["pedigree_id"], ind.metadata["pedigree_p1"], ind.metadata["pedigree_p2"]
            ages[ch] = slim_ts.nodes_time[ind.nodes[0]]
            sex[ch] = ind.metadata["sex"]
            locations[ch] = ind.location
            parents[ch] = [p1, p2]

        # prune individuals with no descendants from the full pedigree
        ancestors = set()
        curr_generation = set(sample_codes)
        while len(curr_generation):
            ancestors.update(curr_generation)
            next_generation = set()
            for child in curr_generation:
                if not np.any(np.equal(parents[child], tskit.NULL)):  # not a founder
                    next_generation.update(parents[child])
            curr_generation = next_generation

        # stick 'em in the pedigree builder
        time_order = sorted(ancestors, key=lambda n: ages[n])
        id_map = {ind:i for i, ind in enumerate(time_order)}
        pedigree = msprime.PedigreeBuilder(individuals_metadata_schema=tskit.MetadataSchema.permissive_json())
        for ind in time_order:
            founder = ages[ind] == args.ticks
            if founder: assert np.all(np.equal(parents[ind], tskit.NULL))
            uid = pedigree.add_individual(
                time=ages[ind], 
                parents=None if founder else [id_map[p] for p in parents[ind]], 
                is_sample=(ages[ind] == 0),
                metadata={
                    "location_x": locations[ind][0], 
                    "location_y": locations[ind][1], 
                    "sex": sex[ind],
                    "pedigree_id": ind,
                },
            )
            assert uid == id_map[ind]
        pedigree_tables = pedigree.finalise(sequence_length=1)
        tszip.compress(pedigree_tables.tree_sequence(), pedigree_tables_path)
    else:
        pedigree_tables = tszip.decompress(pedigree_tables_path).dump_tables()


    # msprime simulation conditional on pedigree
    genealogies_path = args.out_prefix + ".genealogies.trees"
    if not os.path.exists(genealogies_path) or args.overwrite or args.overwrite_from_msp:
        pedigree_tables.sequence_length = args.sequence_length
        pedigree_ts = msprime.sim_ancestry(
            initial_state=pedigree_tables.tree_sequence(),
            recombination_rate=args.recombination_rate,
            model="fixed_pedigree",
            random_seed=args.seed + 1000,
        )

        # recapitate (complete the genealogies) using "empirical" coalescence
        # rate to get population size, over the latter half of the simulation
        # (which hopefully is old enough to avoid spatial structure)
        population_size = 1 / 2 / pedigree_ts.pair_coalescence_rates(
            time_windows=np.array([0, args.ticks / 2, args.ticks, np.inf]),
        )[1]
        logfile.write(f"Recapitating with population size {population_size}\n")
        logfile.write(
            f"Nodes, edges before recapitating: "
            f"{pedigree_ts.num_nodes}, {pedigree_ts.num_edges}\n"
        )
        full_ts = msprime.sim_ancestry(
            initial_state=pedigree_ts,
            population_size=population_size,
            random_seed=args.seed + 2000,
        )
        logfile.write(
            f"Nodes, edges after recapitating: "
            f"{full_ts.num_nodes}, {full_ts.num_edges}\n"
        )
        full_ts = msprime.sim_mutations(
            full_ts, 
            rate=args.mutation_rate,
            random_seed=args.seed + 3000,
        )
        logfile.write(
            f"Simulated {full_ts.num_mutations} mutations with "
            f"mu={args.mutation_rate}\n"
        )
        logfile.write(
            f"Diversity: {full_ts.diversity():.4f}\n"
            f"Segregating sites: {full_ts.segregating_sites():.4f}\n"
        )
        tszip.compress(full_ts, genealogies_path)
    else:
        full_ts = tszip.decompress(genealogies_path)

    
    # trait simulation with tstrait
    # FIXME: no clue what's realistic here
    phenotype_path = args.out_prefix + ".phenotypes.pkl"
    if not os.path.exists(phenotype_path) or args.overwrite or args.overwrite_from_msp:
        phenotypes = tstrait.sim_phenotype(
            full_ts, 
            model=tstrait.trait_model(distribution="normal", mean=0, var=1),
            num_causal=min(args.num_causal_mutations, full_ts.num_mutations),
            #causal_sites=np.arange(full_ts.num_sites),
            h2=args.heritability,
            random_seed=args.seed + 4000,
        )
        pickle.dump(phenotypes, open(phenotype_path, "wb"))
    else:
        phenotypes = pickle.load(open(phenotype_path, "rb"))
    # NB: presumably the phenotypes aren't correct for internal individuals in
    # the pedigree, as only a partial sequence is recorded for these

    # sanity check: we should have both spatial genetic structure
    # and spatial autocorrelation in phenotype, at least at short distances
    stats_path = args.out_prefix + ".stats.pkl"
    if not os.path.exists(stats_path) or args.overwrite or args.overwrite_from_msp:
        sample_individuals = [
            ind.id for ind in full_ts.individuals() if
            np.bitwise_and(full_ts.nodes_flags[ind.nodes[0]], tskit.NODE_IS_SAMPLE)
        ]
        sample_locations = np.array([
            [ind.metadata["location_x"], ind.metadata["location_y"]]
            for ind in full_ts.individuals() if
            np.bitwise_and(full_ts.nodes_flags[ind.nodes[0]], tskit.NODE_IS_SAMPLE)
        ])
        sample_phenotypes = phenotypes.phenotype['phenotype'].to_numpy()[sample_individuals]
        sample_phenotypes = (sample_phenotypes - sample_phenotypes.mean()) / sample_phenotypes.std()

        # doing all pairwise computations is expensive, so for viz subsample a
        # few individuals relatively evenly in space, then calculate stats that
        # are this focal subset vs everyone else
        focal_points = np.array([
            [x, y] 
            for x in np.linspace(0.1, 0.9, 8) 
            for y in np.linspace(0.1, 0.9, 8)
        ]) * args.spatial_extent
        focal_individuals = np.unique([
            np.argmin(np.linalg.norm(sample_locations - coord[None, :], axis=1))
            for coord in focal_points
        ])
        nonfocal_individuals = np.setdiff1d(
            np.arange(len(sample_individuals)),
            focal_individuals,
        )
        focal_samples = np.array(
            [full_ts.individual(i).nodes for i in focal_individuals]
        ).flatten()
        nonfocal_samples = np.array(
            [full_ts.individual(i).nodes for i in nonfocal_individuals]
        ).flatten()

        # various pairwise measures
        spatial_distances = np.sqrt(
            np.subtract.outer(
                sample_locations[focal_individuals, 0], 
                sample_locations[nonfocal_individuals, 0],
            ) ** 2 +
            np.subtract.outer(
                sample_locations[focal_individuals, 1], 
                sample_locations[nonfocal_individuals, 1],
            ) ** 2
        )
        phenotype_distances = \
            np.subtract.outer(
                sample_phenotypes[focal_individuals], 
                sample_phenotypes[nonfocal_individuals],
            )
        genetic_relatedness = full_ts.genetic_relatedness_vector(
            np.eye(full_ts.num_samples)[:, focal_samples], 
            nodes=nonfocal_samples,
            centre=False,
            mode='branch',
        ).T

        # the GRM is a block matrix [[G_ii G_ij], [G_ji G_jj]] where i,j are the
        # "focal" and "nonfocal" samples above, and `genetic_relatedness` is G_ij.
        # so to collapse into individuals,
        sample_to_ind = np.zeros((len(sample_individuals), full_ts.num_samples))
        for i in sample_individuals: 
            sample_to_ind[i][full_ts.individual(i).nodes] = 1.0
        genetic_relatedness = \
            sample_to_ind[np.ix_(focal_individuals, focal_samples)] @ \
            genetic_relatedness @ \
            sample_to_ind[np.ix_(nonfocal_individuals, nonfocal_samples)].T
        pickle.dump(
            (sample_locations, sample_phenotypes, spatial_distances, 
                phenotype_distances, genetic_relatedness),
            open(stats_path, "wb"),
        )
    else:
        sample_locations, sample_phenotypes, spatial_distances, \
            phenotype_distances, genetic_relatedness = \
                pickle.load(open(stats_path, "rb"))


    fig = plt.figure(figsize=(10, 4.5), constrained_layout=True)
    gsp = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gsp[0, 0])
    ax2 = fig.add_subplot(gsp[1, 0])
    ax3 = fig.add_subplot(gsp[:, 1])

    spat = spatial_distances.flatten()
    phen = phenotype_distances.flatten()
    gene = genetic_relatedness.flatten()
    ax1.hexbin(spat[spat < 15], gene[spat < 15], mincnt=1)
    ax1.set_xticks([])
    ax1.set_ylabel("genetic relatedness")
    ax2.hexbin(spat[spat < 15], phen[spat < 15], mincnt=1)
    ax2.set_ylabel("difference in phenotype")
    ax2.set_xlabel("spatial distance between individuals")
    img = ax3.scatter(
        sample_locations[:, 0], 
        sample_locations[:, 1], 
        s=4, 
        c=sample_phenotypes, 
        cmap="terrain",
    )
    plt.colorbar(img, ax=ax3, label="phenotype")
    ax3.set_xlabel("x-coord")
    ax3.set_ylabel("y-coord")
    plt.savefig(f"{args.out_prefix}.png")
