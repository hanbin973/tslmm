import numpy as np
import tskit


def _split_node_up(n, position, ts, tables, nodes_edge, edges_left):
    """
    Move up from node n, splitting any edges above it.
    """
    edge_id = nodes_edge[n]
    while edge_id != tskit.NULL:
        edge = ts.edge(edge_id)
        el = edges_left[edge_id]
        er = position
        if el < er:
            tables.edges.add_row(
                left=el, right=er, parent=edge.parent, child=edge.child
            )
        edges_left[edge_id] = position
        edge_id = nodes_edge[edge.parent]


def split_upwards(ts):
    """
    Returns the tree sequence that is the same as ts except that edges are split so that
    the subtree below any edge does not change.

    (Note: the operation TreeSequence.split() splits edges along a single time.)
    """
    tables = ts.dump_tables()
    tables.edges.clear()
    edges_left = ts.edges_left.copy()
    nodes_edge = np.full((ts.num_nodes,), tskit.NULL)
    for (left, right), edges_out, edges_in in ts.edge_diffs(include_terminal=True):
        for edge in edges_out:
            el = edges_left[edge.id]
            if el < edge.right:
                tables.edges.add_row(
                    left=el, right=edge.right, parent=edge.parent, child=edge.child
                )
            _split_node_up(edge.parent, left, ts, tables, nodes_edge, edges_left)
            nodes_edge[edge.child] = tskit.NULL
        for edge in edges_in:
            _split_node_up(edge.parent, left, ts, tables, nodes_edge, edges_left)
            nodes_edge[edge.child] = edge.id
    tables.sort()
    return tables.tree_sequence()
