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

    If the tree sequence is properly sorted, 
    PmI is upper-triangular.
    """
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

    # iterate over columns
    data_segs = []
    indices_segs = []
    indptr_diff = [0]
    for j in range(IF.shape[1]):
        b = IF[:, j].toarray().ravel()
        x = scipy.sparse.linalg.spsolve_triangular(PmI, b, lower=False)
        i_nnz = np.flatnonzero(x)
        x_nnz = x[i_nnz]
        nnz = x_nnz.size
        data_segs.append(x_nnz)
        indices_segs.append(i_nnz)
        indptr_diff.append(nnz)
    data = np.concatenate(data_segs)
    indices = np.concatenate(indices_segs)
    indptr = np.cumsum(indptr_diff)
    x = scipy.sparse.csc_matrix(
        (data, indices, indptr),
        shape=IF.shape
    )
    xx = x[samples, :]
    R = xx.dot(xx.transpose())
    return R

if __name__ == "__main__":
    ts = tskit.load("test.trees")
    
    samples, = np.where(ts.individuals_time == 0.0)
    samples = samples[:3]
    
    pedigree_matrix(ts, samples, founder_time=3.0)
