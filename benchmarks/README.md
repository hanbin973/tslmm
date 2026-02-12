To use:

1. Produce simulations: once; see below.
    It is then subset (along the genome and by samples) to run benchmarks.

2. Run `do_predictions.sh`, which will run experiments and save results to json files
    in subdirectories.

3. Run `python plot_results.py <name of subdir>` to plot the results saved in a subdirectory;
    plots will go in the subdirectory.


# Simulations:

Desired:

- one populations, but also
- two populations, to give some higher-level structure
- a range of numbers of samples

`one_pop.trees` : `one_pop.py`
    single population, with a bottleneck and a large recent expansion

`two_pop.trees` : `two_pop.py`
    two populations, splitting recently with a bottleneck and large recent expansions

# Speed: `timing.pdf`

How fast is the method, as a function of number of samples and genome length (ie number of trees)?


# Prediction accuracy: `rmse.pdf`

How good are individual-level predictions?
What about edge-level predictions (if we have them)?
How well calibrated are uncertainty estimates?


# Matrix sizes

This also produces a plot studying the results of "splitting upwards": 
see `matrix_sizes.py` and `plot_matrix_sizes.py`.
