import functools

import pytest
import numpy as np
import tskit
import msprime

import tslmm


# NOTE: copied from tskit's tsutil test module. Can delete this if/when we port into
# tskit.
@functools.lru_cache(maxsize=None)
def all_trees_ts(n):
    """
    Generate a tree sequence that corresponds to the lexicographic listing
    of all trees with n leaves (i.e. from tskit.all_trees(n)).

    Note: it would be nice to include a version of this in the combinatorics
    module at some point but the implementation is quite inefficient. Also
    it's not entirely clear that the way we're allocating node times is
    guaranteed to work.
    """
    tables = tskit.TableCollection(0)
    for _ in range(n):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    for j in range(1, n):
        tables.nodes.add_row(flags=0, time=j)

    L = 0
    for tree in tskit.all_trees(n):
        for u in tree.preorder()[1:]:
            tables.edges.add_row(L, L + 1, tree.parent(u), u)
        L += 1
    tables.sequence_length = L
    tables.sort()
    tables.simplify()
    return tables.tree_sequence()


def split_by_tree(ts):
    """
    Naive "wrong" method that splits every edge in every tree, just to make sure
    that we're not doing this in the real code.
    """
    tables = ts.dump_tables()
    tables.edges.clear()
    for tree in ts.trees():
        for u in tree.nodes():
            v = tree.parent(u)
            if v != tskit.NULL:
                tables.edges.add_row(tree.interval.left, tree.interval.right, v, u)
    tables.sort()
    return tables.tree_sequence()


def assert_edge_squash_identical(source_ts, split_ts):
    # should be able to get back the original ts by squashing edges
    split_tables = split_ts.dump_tables()
    split_tables.edges.squash()
    source_ts.tables.assert_equals(split_tables, ignore_provenance=True)


def assert_subtrees_do_not_change(split_ts):
    # check every edge's span contains the span of any parent edges,
    # which suffices to confirm that no subtree changes below any edge
    for t in split_ts.trees():
        for n in t.nodes():
            p = t.parent(n)
            if p != tskit.NULL:
                en = split_ts.edge(t.edge(n))
                ep_id = t.edge(p)
                if ep_id != tskit.NULL:
                    ep = split_ts.edge(ep_id)
                    assert en.left <= ep.left and en.right >= ep.right


class TestSplit:
    def example_ts(self):
        ts = msprime.sim_ancestry(
            100,
            population_size=100,
            sequence_length=100,
            model="dtwf",
            recombination_rate=0.01,
        )
        return ts

    @staticmethod
    def check_split(ts, split_ts):
        assert_edge_squash_identical(ts, split_ts)
        assert_subtrees_do_not_change(split_ts)
        # We need some sort of minimality check here too - how do we know
        # we're not oversplitting??

    def test_large_example(self):
        ts = self.example_ts()
        split_ts = tslmm.split_upwards(ts)
        self.check_split(ts, split_ts)
        split_ts2 = tslmm.split_upwards_numba(ts)
        split_ts.tables.assert_equals(split_ts2.tables, ignore_provenance=True)

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_all_trees_ts(self, n):
        ts = all_trees_ts(n)
        split_ts = tslmm.split_upwards(ts)
        split_ts2 = tslmm.split_upwards_numba(ts)
        # print(split_ts2.draw_text())
        # assert split_ts.num_edges > ts.num_edges
        self.check_split(ts, split_ts)
        split_ts.tables.assert_equals(split_ts2.tables, ignore_provenance=True)

    @pytest.mark.skip("Not currently working")
    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_split_by_tree_fails(self, n):
        ts = all_trees_ts(n)
        print(ts.draw_text())
        print(ts.tables.edges)
        split_ts = tslmm.split_upwards(ts)
        over_split_ts = split_by_tree(ts)
        assert split_ts.num_edges > ts.num_edges
        print(ts.num_edges, split_ts.num_edges, over_split_ts.num_edges)
        assert over_split_ts.num_edges > split_ts.num_edges
        self.check_split(ts, over_split_ts)

    def test_simple_example(self):
        # should split the edges from 3 and 4 up to 5:
        #
        # 2.00┊   5   ┊   5   ┊
        #     ┊  ┏┻━┓ ┊ ┏━┻┓  ┊
        # 1.00┊  3  4 ┊ 3  4  ┊
        #     ┊ ┏┻┓ ┃ ┊ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊ 0 1 2 ┊
        #     0       5      10
        tables = tskit.TableCollection(sequence_length=10)
        tables.nodes.add_row(time=0, flags=1)  # 0
        tables.nodes.add_row(time=0, flags=1)  # 1
        tables.nodes.add_row(time=0, flags=1)  # 2
        tables.nodes.add_row(time=1)  # 3
        tables.nodes.add_row(time=1)  # 4
        tables.nodes.add_row(time=2)  # 5
        tables.edges.add_row(left=0, right=10, parent=3, child=0)
        tables.edges.add_row(left=0, right=5, parent=3, child=1)
        tables.edges.add_row(left=5, right=10, parent=4, child=1)
        tables.edges.add_row(left=0, right=10, parent=4, child=2)
        tables.edges.add_row(left=0, right=10, parent=5, child=3)
        tables.edges.add_row(left=0, right=10, parent=5, child=4)
        ts = tables.tree_sequence()
        split_ts = tslmm.split_upwards(ts)
        self.check_split(ts, split_ts)
        assert ts.num_edges == 6
        assert np.sum(ts.edges_child == 3) == 1
        assert np.sum(ts.edges_child == 4) == 1
        assert split_ts.num_edges == 8
        assert np.sum(split_ts.edges_child == 3) == 2
        assert np.sum(split_ts.edges_child == 4) == 2

        split_ts2 = tslmm.split_upwards_numba(ts)
        split_ts.tables.assert_equals(split_ts2.tables, ignore_provenance=True)
