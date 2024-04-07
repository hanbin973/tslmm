import pytest
import numpy as np
import tskit
import msprime

import tsblup


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
        # should be able to get back the original ts by squashing edges
        split_tables = split_ts.dump_tables()
        split_tables.edges.squash()
        ts.tables.assert_equals(split_tables, ignore_provenance=True)

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

    def test_split(self):
        ts = self.example_ts()
        split_ts = tsblup.split_upwards(ts)
        self.check_split(ts, split_ts)

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
        split_ts = tsblup.split_upwards(ts)
        self.check_split(ts, split_ts)
        assert ts.num_edges == 6
        assert np.sum(ts.edges_child == 3) == 1
        assert np.sum(ts.edges_child == 4) == 1
        assert split_ts.num_edges == 8
        assert np.sum(split_ts.edges_child == 3) == 2
        assert np.sum(split_ts.edges_child == 4) == 2
        print(ts.draw_text())
        print(split_ts.draw_text())
        print(ts.tables.edges)
        print(split_ts.tables.edges)
