import tskit
import tszip
import tsinfer
import os
import numpy as np
import logging
import argparse


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
        "--num-threads", type=int, default=10,
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
