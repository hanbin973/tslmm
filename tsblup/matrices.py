import numpy as np
import scipy.sparse as sparse

import numba

import tskit

# Simulation validation component
def mutation_edge_matrix(ts):
    """
    Returns the (sparse) matrix A for which A[m,e] = 1 if mutation m is on edge e.
    """
    assert ts.num_mutations > 0, "Simulate mutation first"
    return sparse.csr_matrix(
        (
            np.ones(ts.num_mutations),
            (
                np.arange(ts.num_mutations),
                [m.edge for m in ts.mutations()]
            )
        ),
        shape=(ts.num_mutations, ts.num_edges)
    )

# Genetic relatedness matrix components
def edge_child_matrix(ts):
    """
    Returns the (sparse) matrix A for which A[e,u] = 1 if node u is the child of edge e.
    """
    return sparse.csr_matrix(
            (
                np.ones(ts.num_edges),
                (
                    np.arange(ts.num_edges),
                    ts.edges_child,
                ),
            ),
            shape=(ts.num_edges, ts.num_nodes)
    )

def parent_edge_matrix(ts):
    """
    Returns the (sparse) matrix A for which A[u,e] = 1 if edge e is the child of node u.
    """
    return sparse.csr_matrix(
            (
                np.ones(ts.num_edges),
                (
                    ts.edges_parent,
                    np.arange(ts.num_edges),
                ),
            ),
            shape=(ts.num_nodes, ts.num_edges)
    )

def node_adjacency(ts):
    """
    Returns the (sparse) matrix A for which A[u,v] = 1 if there is an edge from u to v.
    """
    A = edge_child_matrix(ts)
    B = parent_edge_matrix(ts)
    return B.dot(A)

@numba.njit
def _edge_adjacency_prune(edges_left, edges_right, num_edges, data, indices, indptr):
    for e in range(num_edges):
        for f_idx in range(indptr[e], indptr[e+1]):
            if (edges_left[e] >= edges_right[indices[f_idx]]) or (edges_left[indices[f_idx]] >= edges_right[e]):
                data[f_idx] = 0

def edge_adjacency(ts):
    """
    Returns the (sparse) matrix A for which A[e,f] = 1 if the child of edge e is the parent of edge f.
    """
    A = edge_child_matrix(ts)
    B = parent_edge_matrix(ts)

    C = A.dot(B)
    _edge_adjacency_prune(
        ts.edges_left, 
        ts.edges_right,
        ts.num_edges,
        C.data,
        C.indices,
        C.indptr
    )
    C.eliminate_zeros()

    return C


# Random-effect design components
def node_individual_matrix(ts):
    """
    Returns the (sparse) matrix A for which A[u,i] = 1 if individual i has node u.
    """
    return sparse.csr_matrix(
            (
                np.ones(ts.num_samples),
                (
                    np.arange(ts.num_samples),
                    ts.nodes_individual[:ts.num_samples]
                )
            ),
            shape=(ts.num_nodes, ts.num_individuals)
    )

def edge_individual_matrix(ts):
    """
    Returns the (sparse) matrix A for which A[e,i] = 0,1,2 is the number of nodes of individual i that are childs of edge e.    
    """
    A = edge_child_matrix(ts)
    B = node_individual_matrix(ts)
    return A.dot(B)
