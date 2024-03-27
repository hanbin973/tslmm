import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba

import scipy.sparse as sparse
import scipy.io as io

import tskit as ts
import tstrait
import msprime
import ldgm

@numba.njit
def _edge_indices_indptr(edges_parent, edges_child, edges_left, edges_right, num_nodes, num_samples):
    """
    Construct edge-edge adjancency matrix ingredients

    :param np.ndarray edges_parent: An array of parent nodes of edges.
    :param np.ndarray edges_child: An array of child nodes of edges.
    :param np.ndarray edges_left: An array of left coordinates of edges.
    :param np.ndarray edges_right: An array of right coordinates of edges.
    :param int num_nodes: Number of samples of tree sequence.
    :param int num_samples: Number of samples of tree sequence.

    :return: index of child bricks and corresponding index pointers
    :rtype: np.ndarray
    """

    # construct pointer to edges with the same parent node
    # idea is adopted from compressed row/column sparse matrices
    nodes_num_edges = np.zeros(num_nodes+1, dtype=np.int32)
    for u_parent in edges_parent:
        nodes_num_edges[u_parent+1] += 1
    edge_ptr = np.cumsum(nodes_num_edges)

    # climb down tree downwards
    # e_parent (number large) --> u_child --> e_child (number small)
    # e_begin <= e_child < e_end
    edges_child_indices = []
    edges_num_childs = np.zeros(edges_parent.shape[0]+1, dtype=np.int32)
    for e_parent, u_child in enumerate(edges_child): 
        e_begin, e_end = edge_ptr[u_child], edge_ptr[u_child+1]
        for e_child in range(e_begin, e_end):
            if (edges_right[e_child] > edges_left[e_parent]) and (edges_right[e_parent] > edges_left[e_child]):
                edges_num_childs[e_parent+1] += 1 
                edges_child_indices.append(e_child)

    return np.asarray(edges_child_indices), np.cumsum(edges_num_childs)

def _make_adjacency(ts):
    indices, indptr = _edge_indices_indptr(
        ts.edges_parent, ts.edges_child, ts.edges_left, ts.edges_right, ts.num_nodes, ts.num_samples
    )
    adjacency = sparse.csr_matrix(
        (
            np.ones(indptr[-1], dtype=np.int32),
            indices,
            indptr
        ),
        shape=(ts.num_edges, ts.num_edges)
    ).T

    return adjacency
