import argparse
import sys
import json

import tskit
import numpy as np
import tslmm

import time

def parse_args():
    dstring = "Simulate and predict phenotypes on a subset of a tree sequence."
    parser = argparse.ArgumentParser(description=dstring,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', type=str,
                        help="Tree sequence filename")
    parser.add_argument('output', type=str,
                        help="Output file base name")
    parser.add_argument('--num_indivs', '-N', type=int, required=True,
                        help="Number of total individuals")
    parser.add_argument('--prop_observed', '-P', type=float, required=True,
                        help="Proportion of individuals with observed phenotypes")
    parser.add_argument('--length', '-L', type=float,
                        help="Genome length")
    parser.add_argument('--heritability', '-H', type=float, required=True,
                        help="proportion of total variance from residuals")
    parser.add_argument('--tau', '-T', type=float, required=True,
                        help="SD of genomic effects (per mutation per generation)")
    parser.add_argument('--pcg_rank', '-R', type=int, default=10,
                        help="Rank of preconditioner")
    parser.add_argument('--num_covariates', '-C', type=int, default=0,
                        help="Number of covariates for simulated traits")
    parser.add_argument('--seed', '-S', type=int, required=True,
                        help="Random seed")
    return parser

def simulate(sigma2, tau2, tree_sequence, mutation_rate, 
             num_covariates=0, rng=None):
    """
    float sigma2: non-genetic variance, i.e., residual
    float tau2: genetic variance
    tskit.TreeSequence tree_sequence: tree sequence
    float mutation_rate: mutation rate (e.g. 1e-8)
    np.random.Generator rng: numpy random number generator
    """
    if rng is None: rng = np.random.default_rng()
    X = rng.normal(size=(tree_sequence.num_individuals, num_covariates)) # covariates
    g = tslmm.simulations.sim_genetic_value(tree_sequence) * np.sqrt(mutation_rate * tau2) # genetic value 
    e = rng.normal(size=tree_sequence.num_individuals) * np.sqrt(sigma2) # residual
    b = rng.normal(size=5) # fixed effect size
    y = g + e # trait value
    if num_covariates > 0:
        y += X @ b
    return y, X, b, g

if __name__ == "__main__":
    mu = 1e-8
    parser = parse_args()
    args = parser.parse_args(sys.argv[1:])

    if args.num_covariates > 0:
        raise ValueError("This does not properly use covariates yet!")

    num_indivs = args.num_indivs
    prop_observed = args.prop_observed
    heritability = args.heritability
    assert heritability >= 0 and heritability <= 1.0
    tau = args.tau      # genomes; NOT squared

    rng = np.random.default_rng(seed=args.seed)

    start_time = time.time()
    ts = tskit.load(args.input).trim()

    L = args.length
    if L is None:
        L = ts.sequence_length

    if L < ts.sequence_length:
        left = rng.choice(int(ts.sequence_length - L))
        ts = ts.keep_intervals([[left, left + L]]).trim()

    sample_indivs = np.array(
            [ind.id for ind in ts.individuals() if ts.node(ind.nodes[0]).is_sample()]
    )
    indivs = rng.choice(sample_indivs, size=num_indivs, replace=False)
    ts = ts.simplify([n for ind in indivs for n in ts.individual(ind).nodes])
    pre_time = time.time() - start_time
    print(f"Preprocessing ts done in {pre_time:.2f} seconds.")

    ind_obs = rng.choice(num_indivs, int(0.9 * num_indivs), replace=False)

    # determine heritability
    start_time = time.time()

    _, _, _, y_bar = simulate(sigma2=0, tau2=tau**2, tree_sequence=ts, mutation_rate=mu,
                              num_covariates=args.num_covariates, rng=rng)
    yv = np.var(y_bar)
    sigma = np.sqrt(yv * (1 - heritability) / heritability)
    y, X, b, y_bar = simulate(sigma2=sigma**2, tau2=tau**2, tree_sequence=ts, mutation_rate=mu,
                              num_covariates=args.num_covariates, rng=rng)
    sim_time = time.time() - start_time
    print(f"Simulation done (twice) in {sim_time:.2f} seconds, with sigma={sigma}.")
    print(f"Trait SD: {np.std(y):.2f}; trait minus genetic value SD: {np.std(y - y_bar):.2f}")

    assert len(y) == num_indivs
    y_obs = y[ind_obs]
    X_obs = X[ind_obs, :]
    lmm = tslmm.TSLMM(ts, mu, y_obs, X_obs, phenotyped_individuals=ind_obs, rng=rng,
                      preconditioner_rank=args.pcg_rank)

    #############
    # use true variance components
    start_time = time.time()

    lmm.set_variance_components([sigma**2, tau**2])
    y_pred, y_var = lmm.predict(np.arange(ts.num_individuals), variance_samples=100)

    pred_time = time.time() - start_time
    print(f"Prediction (once) done in {pred_time:.2f} seconds with 100 variance samples")

    not_obs = np.logical_not(np.isin(np.arange(ts.num_individuals), ind_obs))
    rmse_obs = np.std(y[ind_obs] - y_pred[ind_obs])
    rmse_pred = np.std(y[not_obs] - y_pred[not_obs])
    gen_rmse_obs = np.std(y_bar[ind_obs] - y_pred[ind_obs])
    gen_rmse_pred = np.std(y_bar[not_obs] - y_pred[not_obs])
    se_obs = np.sqrt(y_var)[ind_obs]
    se_pred = np.sqrt(y_var)[not_obs]
    pred_z = rmse_pred / se_pred

    #############
    # infer true variance components
    start_time = time.time()

    lmm.fit_variance_components(verbose=True)

    var_time = time.time() - start_time
    print(f"Variance components fit in {var_time:.2f} seconds")


    results = vars(args)
    results["L"] = L
    results.update({
        "sigma": sigma,
        "y": y.tolist(),
        "y_pred": y_pred.tolist(),
        "y_var": y_var.tolist(),
        "pred_z": pred_z.tolist(),
        "ind_obs": ind_obs.tolist(),
        "pre_time": pre_time,
        "sim_time": sim_time,
        "pred_time": pred_time,
        "var_time": var_time,
        "rmse_obs": rmse_obs,
        "rmse_pred": rmse_pred,
        "gen_rmse_obs": gen_rmse_obs,
        "gen_rmse_pred": gen_rmse_pred,
        "num_edges": ts.num_edges,
    })
    with open(f"{args.output}.json", "w") as f:
        json.dump(results, f)

    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
    ax0.scatter(y[ind_obs], y_pred[ind_obs], label="observed")
    ax0.scatter(y[not_obs], y_pred[not_obs], label="unobserved")
    ax0.set_xlabel("true phenotype")
    ax0.set_ylabel("predicted genetic value")
    ax0.axline((np.mean(y), np.mean(y)), slope=1)

    ax1.scatter(y_bar[ind_obs], y_pred[ind_obs], label="observed")
    ax1.scatter(y_bar[not_obs], y_pred[not_obs], label="unobserved")
    ax1.set_xlabel("true genetic value")
    ax1.set_ylabel("predicted genetic value")
    ax1.axline((np.mean(y), np.mean(y)), slope=1)
    ax1.legend();

    plt.savefig(f"{args.output}.png")
