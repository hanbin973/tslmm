import numpy as np
import scipy.sparse as sparse

from .linalg import *

class IndividualEdgeDesign():
    def __init__(self, Z, T, w):
        """
        Z is the design matrix in CSR
        T is a CSR upper triangular matrix
        w is a vector of edge weights
        """
        self.Z = Z
        self.T = sparse.diags(1/w) @ T
        self.Tp = self.T.indptr
        self.Ti = self.T.indices
        self.Tx = self.T.data
    
    def multiply_left(self, left):
        if left.flags['C_CONTIGUOUS']:
            forward_2dsolve_c(self.Tp, self.Ti, self.Tx, left)
        elif left.flags['F_CONTIGHOUS']:
            forward_2dsolve_f(self.Tp, self.Ti, self.Tx, left)
        else:
            raise ValueError('left is not contiguous')        
        out = self.Z @ left
        out -= out.mean(axis=0)[None,:]
        return out

    def multiply_right(self, right):
        right -= right.mean(axis=0)[None,:]
        out = self.Z.T @ right
        if left.flags['C_CONTIGUOUS']:
            back_2dsolve_c(self.Tp, self.Ti, self.Tx, out)
        elif left.flags['F_CONTIGHOUS']:
            back_2dsolve_f(self.Tp, self.Ti, self.Tx, out)
        else:
            raise ValueError('left is not contiguous')
        return out