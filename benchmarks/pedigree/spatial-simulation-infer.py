import tskit
import tszip
import tsinfer
import os
import json
import numpy as np
import logging
import argparse

import matplotlib.pyplot as plt


docstring = \
"""
- Infer tree sequence from tszip'd spatial simulation
"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser(docstring)
    parser.add_argument(
        "--inp-prefix", type=str, default="spatial-simulation",
        help="Input prefix for tree sequences, etc",
    )
    parser.add_argument(
        "--num-threads", type=int, default=50,
        help="Number of threads for inference",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite all output",
    )
    parser.add_argument(
        "--overwrite-from-tsdate", action="store_true",
        help="Overwrite output from tsdate step onwards",
    )
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.inp_prefix + ".inferred.log",
        level=logging.INFO, 
        filemode="w",
    )
    
    genealogies_path = args.inp_prefix + ".genealogies.trees"
    ts = tszip.decompress(genealogies_path)
    logging.info(f"Input tree sequence:\n{ts}")

    sim_mutations = json.loads(ts.provenance(-1).record)["parameters"]
    assert sim_mutations["command"] == "sim_mutations"
    mutation_rate = sim_mutations["rate"]

    # remove internal pedigree individuals, as tsinfer will choke
    sampled_individuals = \
        np.unique(ts.nodes_individual[list(ts.samples())])
    filter_individuals = np.full(ts.num_individuals, False)
    filter_individuals[sampled_individuals] = True
    tab = ts.dump_tables()
    tab.individuals.packset_parents([[]] * ts.num_individuals)
    individual_map = tab.individuals.keep_rows(filter_individuals)
    tab.nodes.individual = individual_map[tab.nodes.individual]
    tab.sort()
    ts = tab.tree_sequence()
    logging.info(f"Pruned tree sequence:\n{ts}")

    # tsinfer
    inferred_path = args.inp_prefix + ".inferred.trees"
    if not os.path.exists(inferred_path) or args.overwrite:
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
        ts_inf = tsinfer.infer(sample_data, num_threads=args.num_threads)
        tszip.compress(ts_inf, inferred_path)
    else:
        ts_inf = tszip.load(inferred_path)

    # tsdate
    dated_path = args.inp_prefix + ".inferred.dated.trees"
    if not os.path.exists(dated_path) or args.overwrite:
        import tsdate
        ts_date = tsdate.date(
            tsdate.preprocess_ts(ts_inf),
            mutation_rate=mutation_rate,
            rescaling_intervals=20000,
        )
        tszip.compress(ts_date, dated_path)
    else:
        ts_date = tszip.load(dated_path)


    # sanity checks
    fig, axs = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)

    # plot mutation ages, frequency
    sites_mutations = np.bincount(ts_date.mutations_site, minlength=ts_date.num_sites)
    positions_map = {p: i for i, p in enumerate(ts_date.sites_position[sites_mutations == 1])}
    dated_mutation_ages = np.full(len(positions_map), np.nan)
    dated_mutation_freq = np.full(len(positions_map), -1)
    true_mutation_ages = np.full(len(positions_map), np.nan)
    true_mutation_freq = np.full(len(positions_map), -1)

    for t in ts.trees():
        for m in t.mutations():
            if m.edge == tskit.NULL: continue
            p = ts.sites_position[m.site]
            if p in positions_map:
                i = positions_map[p]
                true_mutation_ages[i] = (t.time(t.parent(m.node)) + t.time(m.node)) / 2
                true_mutation_freq[i] = t.num_samples(m.node)

    for t in ts_date.trees():
        for m in t.mutations():
            if m.edge == tskit.NULL: continue
            p = ts_date.sites_position[m.site]
            if p in positions_map:
                i = positions_map[p]
                dated_mutation_ages[i] = (t.time(t.parent(m.node)) + t.time(m.node)) / 2
                dated_mutation_freq[i] = t.num_samples(m.node)

    xm = true_mutation_ages.mean()
    axs[0].hexbin(true_mutation_ages, dated_mutation_ages, xscale="log", yscale="log", mincnt=1)
    axs[0].axline((xm, xm), (xm + 1, xm + 1), color="red", linestyle="dashed")
    axs[0].set_xlabel("true mutation age")
    axs[0].set_ylabel("inferred mutation age")

    # relatedness
    dated_relatedness = ts_date.genetic_relatedness_vector(
        np.eye(ts_date.num_samples)[:, [0]],
        mode='branch',
        centre=False,
    )[1:]
    true_relatedness = ts.genetic_relatedness_vector(
        np.eye(ts.num_samples)[:, [0]],
        mode='branch',
        centre=False,
    )[1:]
    xm = true_relatedness.mean()
    axs[1].hexbin(true_relatedness, dated_relatedness, mincnt=1)
    axs[1].axline((xm, xm), (xm + 1, xm + 1), color="red", linestyle="dashed")
    axs[1].set_xlabel("true branch relatedness")
    axs[1].set_ylabel("inferred branch relatedness")

    plt.savefig(f"{args.inp_prefix}.inferred.png")
    plt.clf()



