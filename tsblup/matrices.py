import numpy as np
import scipy.sparse as sparse

import tskit

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

def edge_adjacency(ts):
    """
    Returns the (sparse) matrix A for which A[e,f] = 1 if the child of edge e is the parent of edge f.
    """
    A = edge_child_matrix(ts)
    B = parent_edge_matrix(ts)
    return A.dot(B)


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
