import numpy as np
import scipy.sparse as sparse

from .linalg_unit import *

class RowEdgeDesign:
    def __init__(self, Z, T, w):
        """
        Z is the design matrix in CSR
        T is a CSR upper triangular matrix
        w is a vector of edge weights
        """
        self.Z = Z
        self.T = T
        self.Tp = self.T.indptr
        self.Ti = self.T.indices
        self.Tx = self.T.data
        self.w = w
        
        self.num_rows = Z.shape[0]
        self.num_edges = Z.shape[1]
    
    def dot_left(self, left):
        left *= self.w[:,None]
        if left.flags['C_CONTIGUOUS']:
            forward_2dsolve_c(self.Tp, self.Ti, self.Tx, left)
        elif left.flags['F_CONTIGUOUS']:
            forward_2dsolve_f(self.Tp, self.Ti, self.Tx, left)
        else:
            raise ValueError('left is not contiguous')        
        out = self.Z @ left
        out -= out.mean(axis=0)[None,:]
        return out

    def dot_right(self, right):
        right -= right.mean(axis=0)[None,:]
        out = self.Z.T @ right
        if out.flags['C_CONTIGUOUS']:
            back_2dsolve_c(self.Tp, self.Ti, self.Tx, out)
        elif out.flags['F_CONTIGHOUS']:
            back_2dsolve_f(self.Tp, self.Ti, self.Tx, out)
        else:
            raise ValueError('out is not contiguous')
        out *= self.w[:,None]
        return out

def randomized_svd(design, n_components=2, n_iter=5, n_oversamples=5, random_matrix=None, seed=None):
    """
    design is the RowEdgeDesign class object
    """
    
    if random_matrix is None:
        if seed is None:
            seed = 0
        np.random.seed(seed)
        random_matrix = np.random.normal(size=(n_components+n_oversamples, design.num_edges)).T

    sample_matrix = design.dot_left(random_matrix)
    range_old, _ = qr(sample_matrix)
    
    for i in range(n_iter):
        range_new, _ = qr(design.dot_right(range_old))
        range_old, _ = qr(design.dot_left(range_new))
        
    U, S, V = np.linalg.svd(design.dot_right(range_old).T, full_matrices=False)
    return (range_old @ U)[:,:n_components], S[:n_components], V[:n_components,:]
    

