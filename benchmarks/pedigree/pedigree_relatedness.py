import tskit
import numpy as np
import scipy.sparse


def pedigree_matrix(ts, samples, founder_time):
    """
    Let x be the matrix whose i,j-th element gives the expected amount
    that individual i inherited from founder-individual j;
    where none of the founders are ancestral to others.
    Then with P[a,b] = 1/2 if b is a parent of a, we have that
       Px[i,j] = x[i,j]  for i not a founder, and
       Px[f,j] = 0 for f a founder;
       x[f,f] = 1 for f a founder,
       x[f,j] = 0 otherwise
    and so
       (P - I) x = -I_F
    where I_F is the identity-for-the-founders matrix
    """
    founder_time = 3.0
    founders, = np.where(ts.individuals_time == founder_time)
    individuals_parent = ts.tables.individuals.parents
    nzp, = np.where(
            np.repeat(ts.individuals_time < founder_time, 2)
    )
    PmI = scipy.sparse.coo_matrix(
            (np.full(len(nzp), 0.5),
             (np.repeat(np.arange(ts.num_individuals), 2)[nzp],
              individuals_parent[nzp])),
            shape=(ts.num_individuals, ts.num_individuals)
    ) - scipy.sparse.eye(ts.num_individuals)
    IF = scipy.sparse.coo_matrix(
            (np.full(len(founders), -1.0),
             (founders, np.arange(len(founders)))),
            (ts.num_individuals, len(founders))
    ).tocsc()
    x = scipy.sparse.linalg.spsolve(PmI, IF)
    xx = x[samples, :]
    assert np.allclose(np.sum(xx, axis=1), 1.0)
    R = xx.dot(xx.transpose())
    return R

ts = tskit.load("test.trees")

samples, = np.where(ts.individuals_time == 0.0)
samples = samples[:3]

pedigree_matrix(ts, samples, founder_time=3.0)
