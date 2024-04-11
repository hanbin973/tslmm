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

# Node approximation
@numba.njit
def _node_transmission_matrix(edges_parent, edges_child, edges_left, edges_right, nodes_time, num_nodes):
    # sparse matrix-styled index pointer to edges
    parent_num_edges = np.zeros(num_nodes+1, dtype=np.int32)
    for parent in edges_parent:
        parent_num_edges[parent+1] += 1
    edgeptr = np.cumsum(parent_num_edges)

    # edge area
    edges_area = (edges_right - edges_left) * (nodes_time[edges_parent] - nodes_time[edges_child])
    
    # construct (sparse) node transmission matrix A
    # eps that weights variances
    values = [np.float64(x) for x in range(0)]
    rows = [np.int32(x) for x in range(0)]
    cols = [np.int32(x) for x in range(0)]
    eps = np.zeros(num_nodes)
    for parent in range(num_nodes):
        e_begin, e_end = edgeptr[parent], edgeptr[parent+1]
        if e_end > e_begin:
            parent_left, parent_right = np.zeros(e_end-e_begin), np.zeros(e_end-e_begin)
            for i, e in enumerate(range(e_begin, e_end)):
                parent_left[i] = edges_left[e]
                parent_right[i] = edges_right[e]
            parent_span = parent_right.max() - parent_left.min()
            for e in range(e_begin, e_end):
                values.append((edges_right[e] - edges_left[e])/parent_span) # divided by total value
                rows.append(edges_child[e])
                cols.append(parent)
                eps[edges_child[e]] += edges_area[e]

    return values, rows, cols, eps

def node_transmission_matrix(ts):
    values, rows, cols, eps = _node_transmission_matrix(
        ts.edges_parent,
        ts.edges_child,
        ts.edges_left,
        ts.edges_right,
        ts.nodes_time,
        ts.num_nodes
    )
    return sparse.csr_matrix(
        (
            values,
            (
                rows,
                cols
            )
        ),
        shape=(ts.num_nodes, ts.num_nodes)
    ), eps
