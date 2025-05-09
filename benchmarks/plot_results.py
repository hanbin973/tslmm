import sys
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys, os

assert len(sys.argv) == 2, "Usage: python plot_results.py <directory name>"
basedir = sys.argv[1]

jj = []
for jf in glob.glob(os.path.join(basedir, "*.json")):
    with open(jf, "r") as f:
        jj.append(json.load(f))

df = pd.DataFrame(jj)

########
pcg_rank_markers = {a : b for a, b in zip(np.unique(df['pcg_rank']), ['o', '^', 's', 'P'])}
# why does mpl hate making interpretable plots
num_indivs_norm = matplotlib.colors.Normalize(vmin=np.min(df.num_indivs), vmax=np.max(df.num_indivs))
num_edges_norm = matplotlib.colors.Normalize(vmin=np.min(df.num_edges), vmax=np.max(df.num_edges))
num_indivs_cmap = plt.colormaps["viridis"]
num_edges_cmap = plt.colormaps["plasma"]

def num_indivs_c(x):
    return num_indivs_cmap(num_indivs_norm(x))

def num_edges_c(x):
    return num_edges_cmap(num_edges_norm(x))

def num_edges_colorbar(ax):
    do_colorbar(ax, num_edges_norm, num_edges_cmap, "number of edges")

def num_indivs_colorbar(ax):
    do_colorbar(ax, num_indivs_norm, num_indivs_cmap, "number of individuals")

def do_colorbar(ax, norm, cmap, label):
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=ax, label=label)


# RMSE plots
fname = os.path.join(basedir, "rmse.pdf")
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 5), sharey=True)

ax0.set_xlabel("number of individuals")
ax1.set_xlabel("number of edges")
ax0.set_ylabel("RMSE genetic value, observed")
ax1.set_ylabel("RMSE genetic value, predicted")
s0 = ax0.scatter(df.num_indivs, df.gen_rmse_obs, marker=".", c=df.heritability, label="observed")
ax0.scatter(df.num_indivs, df.gen_rmse_pred, marker="o", c=df.heritability, label="unobserved")
ax1.scatter(df.num_edges, df.gen_rmse_obs, marker=".", c=df.heritability)
ax1.scatter(df.num_edges, df.gen_rmse_pred, marker="o", c=df.heritability)

ax0.legend()
fig.colorbar(s0, label='heritability')

plt.tight_layout()
plt.savefig(fname)


# timing plots
fname = os.path.join(basedir, "timing.pdf")
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(9, 9))

ax0.set_xlabel("number of edges")
ax0.set_ylabel("prediction runtime (s)")
ax1.set_xlabel("number of individuals")
ax1.set_ylabel("prediction runtime (s)")
for rx, sub_df in df.groupby('pcg_rank'):
    ax0.scatter(sub_df.num_edges, sub_df.pred_time,
                c=num_indivs_c(sub_df.num_indivs),
                marker=pcg_rank_markers[rx],
                label=f"PCG rank={rx}")
    ax1.scatter(sub_df.num_indivs, sub_df.pred_time,
                c=num_edges_c(sub_df.num_edges),
                marker=pcg_rank_markers[rx],
                label=f"PCG rank={rx}")

num_indivs_colorbar(ax0)
ax1.legend()
num_edges_colorbar(ax1)

ax2.set_xlabel("number of edges")
ax2.set_ylabel("variance components runtime (s)")
ax3.set_xlabel("number of individuals")
ax3.set_ylabel("variance components runtime (s)")
ax2.scatter(sub_df.num_edges, sub_df.var_time,
            c=num_indivs_c(sub_df.num_indivs))
ax3.scatter(sub_df.num_indivs, sub_df.var_time,
            c=num_edges_c(sub_df.num_edges))

num_indivs_colorbar(ax2)
num_edges_colorbar(ax3)


plt.tight_layout()
plt.savefig(fname)
