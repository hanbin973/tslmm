import matplotlib.pyplot as plt
import numpy as np
from trace_estimators import hutchinson, xtrace, xnystrace, xdiag

if __name__ == "__main__":

    grid = np.arange(2, 200)
    # TODO use one of our actual matrices
    foo = np.abs(np.random.randn(1000, 1000))
    foo = foo @ foo.T / 2 + foo.T @ foo / 2

    # hutchinson
    uno = np.sum(foo.diagonal())
    strm = [hutchinson(foo, i) for i in grid]
    est = np.array([s for s, e in strm])
    err = np.array([e for s, e in strm])
    print("Last est", uno, est[-1])
    print("95Cov", np.sum(np.logical_and(uno > est - 1.96 * err, uno < est + 1.96 * err)) / grid.size)
    
    plt.scatter(grid, est, s=4, c="red")
    plt.vlines(grid, est - 1.96 * err, est + 1.96 * err, color="red")
    plt.axhline(y=uno, linestyle="--", color="black")
    plt.savefig("figs/hutchinson_convergence.png")
    plt.clf()

    # xtrace
    uno = np.sum(foo.diagonal())
    strm = [xtrace(foo, i) for i in grid]
    est = np.array([s for s, e in strm])
    err = np.array([e for s, e in strm])
    print("Last est", uno, est[-1])
    print("95Cov", np.sum(np.logical_and(uno > est - 1.96 * err, uno < est + 1.96 * err)) / grid.size)
    
    plt.scatter(grid, est, s=4, c="red")
    plt.vlines(grid, est - 1.96 * err, est + 1.96 * err, color="red")
    plt.axhline(y=uno, linestyle="--", color="black")
    plt.savefig("figs/xtrace_convergence.png")
    plt.clf()

    # xnystrace
    uno = np.sum(foo.diagonal())
    strm = [xnystrace(foo, i) for i in grid]
    est = np.array([s for s, e in strm])
    err = np.array([e for s, e in strm])
    print("Last est", uno, est[-1])
    print("95Cov", np.sum(np.logical_and(uno > est - 1.96 * err, uno < est + 1.96 * err)) / grid.size)
    
    plt.scatter(grid, est, s=4, c="red")
    plt.vlines(grid, est - 1.96 * err, est + 1.96 * err, color="red")
    plt.axhline(y=uno, linestyle="--", color="black")
    plt.savefig("figs/xnystrace_convergence.png")
    plt.clf()

    # xdiag
    uno = foo.diagonal()[0]
    strm = [xdiag(foo, i)[0] for i in grid]
    print(uno, strm[-1])
    
    plt.scatter(grid, strm / uno)
    plt.axhline(y=1.0, linestyle="--", color="black")
    plt.savefig("figs/xdiag_convergence.png")
    plt.clf()
