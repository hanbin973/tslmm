import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

params = {
    'font.size': 10,
}
matplotlib.rcParams.update(params)

orig_df = pd.read_csv("matrix_sizes.tsv", sep="\t")
df = orig_df[~orig_df.split].merge(orig_df[orig_df.split], on="nodes", suffixes=["", "_split"], how='outer')


####
outfile = "matrix_sizes.pdf"
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(6.5, 2.5))

ax0.set_xlabel("original number of edges")
ax0.set_ylabel("split increase in number")
ax0.scatter(df["num_edges"], df["num_edges_split"]/df["num_edges"])

ax1.set_xlabel("original number of edges")
ax1.set_ylabel("nonzeros in E")
ax1.scatter(df["num_edges"], df["edge_adj_nnz"], label="original")
ax1.scatter(df["num_edges"], df["edge_adj_nnz_split"], label="split")
ax1.set_yscale("log")
ax1.legend()

ax2.set_xlabel("original number of edges")
ax2.set_ylabel("runtime (s)")
ax2.scatter(df["num_edges"], df["A_dot_B_time_split"], label="P(Cx)")
ax2.scatter(df["num_edges"], df["AB_time_split"], label="Ex=(PC)x")
ax2.set_yscale("log")
ax2.legend()

plt.tight_layout()
plt.savefig(outfile, dpi=300)
