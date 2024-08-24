import tskit
import tslmm
import numpy as np
import scipy.sparse
import timeit

rng = np.random.default_rng(seed=123)

ts = tskit.load("one_pop.trees")
outname = "matrix_sizes.tsv"
outfile = open(outname, "w")
print("\t".join([
    "split",
    "num_edges",
    "nodes",
    "edge_child_nnz",
    "parent_edge_nnz",
    "edge_adj_nnz",
    "A_dot_B_time",
    "AB_time",
]), file=outfile)

for split in [False, True]:
    for L in np.linspace(0.1e6, 0.5e7, 20):
        print(":::", split, int(L))
        sts = ts.keep_intervals([[int(ts.sequence_length/2 - L/2), int(ts.sequence_length/2 + L/2)],]).trim()
        if split:
            sts = tslmm.split_upwards(sts)
        A = tslmm.edge_child_matrix(sts)
        B = tslmm.parent_edge_matrix(sts)
        AB = A.dot(B)
        x = rng.normal(size=B.shape[1])
        A_B_time = timeit.timeit(lambda: A.dot(B.dot(x)), number=20)
        AB_time = timeit.timeit(lambda: AB.dot(x), number=20)
        print("\t".join(map(str, [
            split,
            sts.num_edges,
            sts.num_nodes,
            len(A.data),
            len(B.data),
            len(AB.data),
            A_B_time,
            AB_time,
        ])), file=outfile)
        outfile.flush()


