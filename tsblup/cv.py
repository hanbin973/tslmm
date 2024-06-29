import numpy as np

import scipy
import scipy.sparse as sparse
from scipy.sparse.linalg import LinearOperator

import numba
from numba import types, typed
from numba.types import f8, i4

from functools import partial

from .preconditioner import *

def grm_v(v, design, id_subset, geom, edges_weight):
    w = design.tdot(v[:,None], id_subset)
    geom.back_solve(w)
    w *= edges_weight[:,None]
    geom.forward_solve(w)
    return design.dot(w, id_subset).ravel()

class KernelMatrix():
    def __init__(
        self,
        design_list,
        geom_list,
        edges_weight_list,
        sigma_ep,
        id_subset
    ):
        assert len(design_list) == len(geom_list)
        assert len(geom_list) == len(edges_weight_list)
        self.design_list = design_list
        self.geom_list = geom_list
        self.edges_weight_list = edges_weight_list
        self.sigma_ep = sigma_ep
        self.id_subset = id_subset

    def dot(self, v):
        w = np.zeros(v.size)
        for i in range(len(self.design_list)):
            w += grm_v(
                v,
                self.design_list[i],
                self.id_subset,
                self.geom_list[i],
                self.edges_weight_list[i]
            )
        return w + self.sigma_ep * v

def covyu_v(v, design, id_subset, geom, edges_weight):
    w = design.tdot(v[:,None], id_subset)
    geom.back_solve(w)
    w *= edges_weight[:,None]
    geom.forward_solve(w)
    return w.ravel()

class CovOutCumMatrix():
    def __init__(
        self,
        design_list,
        geom_list,
        edges_weight_list,
        id_subset
    ):
        assert len(design_list) == len(geom_list)
        assert len(geom_list) == len(edges_weight_list)
        self.design_list = design_list
        self.geom_list = geom_list
        self.edges_weight_list = edges_weight_list
        self.id_subset = id_subset

    def dot(self, v):
        w = np.empty(sum([x.size for x in self.edges_weight_list]))
        cnt = 0
        for i in range(len(self.design_list)):
            num_edges = self.edges_weight_list[i].size
            il, iu = cnt, cnt+num_edges
            w[il:iu] = covyu_v(
                v,
                self.design_list[i],
                self.id_subset,
                self.geom_list[i],
                self.edges_weight_list[i]
            )
            cnt += iu
        return w

from scipy.sparse.linalg import LinearOperator

def chunk_array(arr, n_chunk):
    chunk_size = int(arr.size / n_chunk)
    return [arr[i:i+chunk_size] for i in range(0, arr.shape[0], chunk_size)]     

class BLUPCrossValidation:
    def __init__(
        self,
        design_list,
        geom_list,
        edges_area_list,
        y
    ):
        self.design_list = design_list
        self.geom_list = geom_list
        self.edges_area_list = edges_area_list
        self.y = y

        self.n_inds = y.size
        self.n_cv = None

        self.design = Design(
            self.design_list,
            self.geom_list,
            self.edges_area_list,
            self.n_inds
        )

    def set_cv(self, cv=5, n_components=500, n_iter=3):
        # chunk array
        id_inds = np.arange(self.n_inds, dtype=np.int32)
        self.id_chunks = chunk_array(id_inds, cv)

        # construct preconditioner
        self.preconditioners = []
        for id_subset in self.id_chunks:
            self.preconditioners.append(
                RandNystromPreconditioner(
                    self.design,
                    id_subset=np.delete(id_inds, id_subset),
                    n_components=n_components,
                    n_iter=n_iter
                )
            )

        # set cv
        self.n_cv = cv

    def blup_subset(self, id_subset, lam, preconditioner=None):           
        kernel = KernelMatrix(
                self.design_list,
                self.geom_list,
                self.edges_area_list,
                lam,
                id_subset
            )
        kernel_op = LinearOperator(
            (id_subset.size, id_subset.size), 
            matvec=kernel.dot
        )
        covyu = CovOutCumMatrix(
                self.design_list,
                self.geom_list,
                self.edges_area_list,
                id_subset
            )

        if preconditioner is None:
            v1 = sparse.linalg.cg(kernel_op, self.y[id_subset])[0]
            v2 = sparse.linalg.cg(kernel_op, v1)[0]
        else:
            preconditioner_dot = partial(
                preconditioner.dot,
                lam=lam
            )
            preconditioner_op = LinearOperator(
                (id_subset.size, id_subset.size), 
                matvec=preconditioner_dot
            )
            v1 = sparse.linalg.cg(
                kernel_op, 
                self.y[id_subset],
                M=preconditioner_op
            )[0]
            v2 = sparse.linalg.cg(
                kernel_op, 
                v1,
                M=preconditioner_op
            )[0]
        
        return covyu.dot(v1), covyu.dot(v2) 

    def blup(self, lam, id_preconditioner=None):
        return self.compute_base_subset(
            np.arange(self.n_inds, dtype=np.int32), 
            lam,
            preconditioner=None
        )
    
    def cv_error(self, lam):
        assert self.n_cv is not None
        err_nonnormalized, grad = 0, 0
        id_inds = np.arange(self.n_inds, dtype=np.int32)
        for i, id_chunk in enumerate(self.id_chunks):
            id_train, id_test = np.delete(id_inds, id_chunk), id_chunk
            u_blup, gu_blup = self.blup_subset(
                id_train, 
                lam, 
                self.preconditioners[i]
            )            
            y_blup, gy_blup = np.zeros(id_test.size), np.zeros(id_test.size)
            for design in self.design_list:
                y_blup += design.dot(u_blup[:,None], id_test).ravel()
                gy_blup += design.dot(gu_blup[:,None], id_test).ravel()
            residual = self.y[id_test] - y_blup
            err_nonnormalized += np.sum(residual**2)
            grad += np.dot(residual, gy_blup)
        
        return err_nonnormalized / self.n_inds, 2 * grad / self.n_inds
        
class BLUPRandPred:
    def __init__(
        self,
        design_list,
        geom_list,
        edges_area_list,
        y
    ):
        self.design_list = design_list
        self.geom_list = geom_list
        self.edges_area_list = edges_area_list
        self.y = y

        self.n_inds = y.size

        self.design = Design(
            self.design_list,
            self.geom_list,
            self.edges_area_list,
            self.n_inds
        )

    def set_random(self, id_train, n_components=500, n_iter=3):
        # define train/test sets
        id_inds = np.arange(self.n_inds, dtype=np.int32)
        self.id_train, self.id_test = id_train, np.delete(id_inds, id_train)
        
        # construct preconditioner
        self.preconditioner = RandNystromPreconditioner(
            self.design,
            id_subset=id_train,
            n_components=n_components,
            n_iter=n_iter
        )
        
    def blup_subset(self, id_subset, lam, preconditioner=None):           
        kernel = KernelMatrix(
                self.design_list,
                self.geom_list,
                self.edges_area_list,
                lam,
                id_subset
            )
        kernel_op = LinearOperator(
            (id_subset.size, id_subset.size), 
            matvec=kernel.dot
        )
        covyu = CovOutCumMatrix(
                self.design_list,
                self.geom_list,
                self.edges_area_list,
                id_subset
            )

        if preconditioner is None:
            v1 = sparse.linalg.cg(kernel_op, self.y[id_subset])[0]
            v2 = sparse.linalg.cg(kernel_op, v1)[0]
        else:
            preconditioner_dot = partial(
                preconditioner.dot,
                lam=lam
            )
            preconditioner_op = LinearOperator(
                (id_subset.size, id_subset.size), 
                matvec=preconditioner_dot
            )
            v1 = sparse.linalg.cg(
                kernel_op, 
                self.y[id_subset],
                M=preconditioner_op
            )[0]
            v2 = sparse.linalg.cg(
                kernel_op, 
                v1,
                M=preconditioner_op
            )[0]
        
        return covyu.dot(v1), covyu.dot(v2) 

    def blup(self, lam, id_preconditioner=None):
        return self.compute_base_subset(
            np.arange(self.n_inds, dtype=np.int32), 
            lam,
            preconditioner=None
        )
    
    def cv_error(self, loglam):
        u_blup, gu_blup = self.blup_subset(
            self.id_train, 
            np.exp(loglam), 
            self.preconditioner
        )            
        y_blup, gy_blup = np.zeros(self.id_test.size), np.zeros(self.id_test.size)
        for design in self.design_list:
            y_blup += design.dot(u_blup[:,None], self.id_test).ravel()
            gy_blup += design.dot(gu_blup[:,None], self.id_test).ravel()
        residual = self.y[self.id_test] - y_blup
        err_nonnormalized = np.sum(residual**2)
        grad = np.dot(residual, gy_blup) * np.exp(loglam)
        
        return err_nonnormalized / self.n_inds, 2 * grad / self.n_inds
        
        return err_nonnormalized / self.n_inds, 2 * grad / self.n_inds