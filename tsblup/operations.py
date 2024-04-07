import numba
import numpy as np
import tskit

spec = [
    ("num_edges", numba.int64),
    ("sequence_length", numba.float64),
    ("edges_left", numba.float64[:]),
    ("edges_right", numba.float64[:]),
    ("edge_insertion_order", numba.int32[:]),
    ("edge_removal_order", numba.int32[:]),
    ("edge_insertion_index", numba.int64),
    ("edge_removal_index", numba.int64),
    ("interval", numba.float64[:]),
    ("in_range", numba.int64[:]),
    ("out_range", numba.int64[:]),
]


@numba.experimental.jitclass(spec)
class TreePosition:
    def __init__(
        self,
        num_edges,
        sequence_length,
        edges_left,
        edges_right,
        edge_insertion_order,
        edge_removal_order,
    ):
        self.num_edges = num_edges
        self.sequence_length = sequence_length
        self.edges_left = edges_left
        self.edges_right = edges_right
        self.edge_insertion_order = edge_insertion_order
        self.edge_removal_order = edge_removal_order
        self.edge_insertion_index = 0
        self.edge_removal_index = 0
        self.interval = np.zeros(2)
        self.in_range = np.zeros(2, dtype=np.int64)
        self.out_range = np.zeros(2, dtype=np.int64)

    def next(self):  # noqa
        left = self.interval[1]
        j = self.in_range[1]
        k = self.out_range[1]
        self.in_range[0] = j
        self.out_range[0] = k
        M = self.num_edges
        edges_left = self.edges_left
        edges_right = self.edges_right
        out_order = self.edge_removal_order
        in_order = self.edge_insertion_order

        while k < M and edges_right[out_order[k]] == left:
            k += 1
        while j < M and edges_left[in_order[j]] == left:
            j += 1
        self.out_range[1] = k
        self.in_range[1] = j

        right = self.sequence_length
        if j < M:
            right = min(right, edges_left[in_order[j]])
        if k < M:
            right = min(right, edges_right[out_order[k]])
        self.interval[:] = [left, right]
        return j < M or left < self.sequence_length


spec = [
    ("left", numba.types.ListType(numba.types.float64)),
    ("right", numba.types.ListType(numba.types.float64)),
    ("parent", numba.types.ListType(numba.types.int32)),
    ("child", numba.types.ListType(numba.types.int32)),
]


@numba.experimental.jitclass(spec)
class EdgeTable:
    def __init__(self):
        self.left = numba.typed.List.empty_list(numba.types.float64)
        self.right = numba.typed.List.empty_list(numba.types.float64)
        self.parent = numba.typed.List.empty_list(numba.types.int32)
        self.child = numba.typed.List.empty_list(numba.types.int32)

    def add_row(self, left, right, parent, child):
        self.left.append(left)
        self.right.append(right)
        self.parent.append(parent)
        self.child.append(child)


# Helper function to make it easier to create the numba class
def alloc_tree_position(ts):
    return TreePosition(
        num_edges=ts.num_edges,
        sequence_length=ts.sequence_length,
        edges_left=ts.edges_left,
        edges_right=ts.edges_right,
        edge_insertion_order=ts.indexes_edge_insertion_order,
        edge_removal_order=ts.indexes_edge_removal_order,
    )


def split_upwards_numba(ts):
    out_edges = EdgeTable()
    _split_upwards_numba(
        alloc_tree_position(ts),
        ts.num_nodes,
        ts.edges_left.copy(),
        ts.edges_right,
        ts.edges_child,
        ts.edges_parent,
        out_edges,
    )

    tables = ts.dump_tables()
    tables.edges.set_columns(
        left=out_edges.left,
        right=out_edges.right,
        parent=out_edges.parent,
        child=out_edges.child,
    )
    tables.sort()
    return tables.tree_sequence()


@numba.njit
def _split_upwards_numba(
    tree_pos, num_nodes, edges_left, edges_right, edges_child, edges_parent, out_edges
):
    nodes_edge = np.full((num_nodes), tskit.NULL)

    while True:
        tree_pos.next()
        left, right = tree_pos.interval
        for j in range(tree_pos.out_range[0], tree_pos.out_range[1]):
            e = tree_pos.edge_removal_order[j]
            c = edges_child[e]
            p = edges_parent[e]
            el = edges_left[e]

            if el < edges_right[e]:
                out_edges.add_row(el, edges_right[e], p, c)
            _split_node_up2(
                p,
                left,
                out_edges,
                nodes_edge,
                edges_left,
                edges_right,
                edges_parent,
                edges_child,
            )
            nodes_edge[c] = tskit.NULL

        if left == right:
            break

        for j in range(tree_pos.in_range[0], tree_pos.in_range[1]):
            e = tree_pos.edge_insertion_order[j]
            p = edges_parent[e]
            c = edges_child[e]

            _split_node_up2(
                p,
                left,
                out_edges,
                nodes_edge,
                edges_left,
                edges_right,
                edges_parent,
                edges_child,
            )
            nodes_edge[c] = e


@numba.njit
def _split_node_up2(
    u,
    position,
    out_edges,
    nodes_edge,
    edges_left,
    edges_right,
    edges_parent,
    edges_child,
):
    """
    Move up from node u, splitting any edges above it.
    """
    edge_id = nodes_edge[u]
    while edge_id != tskit.NULL:
        el = edges_left[edge_id]
        er = position
        if el < er:
            out_edges.add_row(el, er, edges_parent[edge_id], edges_child[edge_id])
        edges_left[edge_id] = position
        edge_id = nodes_edge[edges_parent[edge_id]]


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
