import numpy as np
import pandas as pd

import tskit
import tstrait
import msprime

import numba
from numba import i4, f8
from numba.experimental import jitclass

class GeneticValueSimulator:
    def __init__(self, ts, beta):
        assert ts.sequence_length == beta.shape[0], "Length of beta should match the sequence length"
        self.ts = ts
        self.beta = beta

    def sim_mutations(self, **kwargs):
        self.mts = msprime.sim_mutations(self.ts, **kwargs)
        
    def sim_genetic_value(self, ignore_parent=False, **kwargs):
        self.sim_mutations(**kwargs)

        effect_size = self.beta[self.mts.sites_position[self.mts.mutations_site].astype(np.int32)]
        if ignore_parent:
            mutations_parent = self.mts.mutations_parent 
            has_parent = np.where(mutations_parent != -1)[0]
            is_parent = np.unique(mutations_parent[has_parent])
            effect_size[is_parent] = 0

        trait_dict = {
            "position": self.mts.sites_position[self.mts.mutations_site],
            "site_id": self.mts.mutations_site,
            "edge_id": [m.edge for m in self.mts.mutations()],
            "effect_size": effect_size,
            "trait_id": np.zeros(self.mts.num_mutations, dtype=np.int32),
            "causal_allele": [m.derived_state for m in self.mts.mutations()]
        }
        self.trait_df = pd.DataFrame(trait_dict)
        self.genetic_df = tstrait.genetic_value(self.mts, self.trait_df)
        

# nubma implementation
spec = [
        ('sample_weights', f8[:]),
        ('parent', i4[:]),
        ('num_samples', i4[:]),
        ('edges_left', f8[:]),
        ('edges_right', f8[:]),
        ('edges_parent', i4[:]),
        ('edges_child', i4[:]),
        ('edge_insertion_order', i4[:]),
        ('edge_removal_order', i4[:]),
        ('sequence_length', f8),
        ('nodes_time', f8[:]),
        ('samples', i4[:]),
        ('position', f8),
        ('virtual_root', i4),
        ('x', f8[:]),
        ('w', f8[:]),
        ('stack', f8[:]),
        ('NULL', i4)
       ] 

@jitclass(spec)
class TraitVector:
    def __init__(
        self,
        num_nodes,
        samples,
        nodes_time,
        edges_left,
        edges_right,
        edges_parent,
        edges_child,
        edge_insertion_order,
        edge_removal_order,
        sequence_length
    ):
        # virtual root is at num_nodes; virtual samples are beyond that
        N = num_nodes + 1 + len(samples)
        # Quintuply linked tree
        self.parent = np.full(N, -1, dtype=np.int32)
        # Sample lists refer to sample *index*
        self.num_samples = np.full(N, 0, dtype=np.int32)
        # Edges and indexes
        self.edges_left = edges_left
        self.edges_right = edges_right
        self.edges_parent = edges_parent
        self.edges_child = edges_child
        self.edge_insertion_order = edge_insertion_order
        self.edge_removal_order = edge_removal_order
        self.sequence_length = sequence_length
        self.nodes_time = nodes_time
        self.samples = samples
        self.position = 0
        self.virtual_root = num_nodes
        self.x = np.zeros(N, dtype=np.float64)
        self.stack = np.zeros(N, dtype=np.float64)
        self.NULL = -1 # to avoid tskit.NULL in numba

        for j, u in enumerate(samples):
            self.num_samples[u] = 1
            # Add branch to the virtual sample
            v = num_nodes + 1 + j
            self.parent[v] = u
            self.num_samples[v] = 1

    def remove_edge(self, p, c):
        self.stack[c] += self.get_z(c)
        self.x[c] = self.position
        self.parent[c] = -1
        self.adjust_path_up(c, p, -1)

    def insert_edge(self, p, c):
        self.adjust_path_up(c, p, +1)
        self.x[c] = self.position
        self.parent[c] = p

    def adjust_path_up(self, c, p, sign):
        # sign = -1 for removing edges, +1 for adding
        while p != self.NULL:
            self.stack[p] += self.get_z(p)
            self.x[p] = self.position
            # check for floating point error
            prev_stack = self.stack[c]
            self.stack[c] -= sign * self.stack[p]
            p = self.parent[p]

    def get_z(self, u):
        p = self.parent[u]
        if p == self.NULL or u >= self.virtual_root:
            return 0.0
        time = self.nodes_time[p] - self.nodes_time[u]
        span = self.position - self.x[u]
        return np.sqrt(time * span) * np.random.normal()

    def run(self):
        sequence_length = self.sequence_length
        M = self.edges_left.shape[0]
        in_order = self.edge_insertion_order
        out_order = self.edge_removal_order
        edges_left = self.edges_left
        edges_right = self.edges_right
        edges_parent = self.edges_parent
        edges_child = self.edges_child

        j = 0
        k = 0
        # TODO: self.position is redundant with left
        left = 0
        self.position = left

        while k < M and left <= self.sequence_length:
            while k < M and edges_right[out_order[k]] == left:
                p = edges_parent[out_order[k]]
                c = edges_child[out_order[k]]
                self.remove_edge(p, c)
                k += 1
            while j < M and edges_left[in_order[j]] == left:
                p = edges_parent[in_order[j]]
                c = edges_child[in_order[j]]
                self.insert_edge(p, c)
                j += 1
            right = sequence_length
            if j < M:
                right = min(right, edges_left[in_order[j]])
            if k < M:
                right = min(right, edges_right[out_order[k]])
            left = right
            self.position = left

        # clear remaining things down to virtual samples
        for j, u in enumerate(self.samples):
            v = self.virtual_root + 1 + j
            self.remove_edge(u, v)

        out = np.zeros(len(self.samples))
        for out_i in range(len(self.samples)):
            i = out_i + self.virtual_root + 1
            out[out_i] = self.stack[i]
        return out



def sim_genetic_value(ts, **kwargs):
    def bincount_fn(w):
        return np.bincount(samples_individuals, w)
    rv = TraitVector(
        ts.num_nodes,
        samples=ts.samples(),
        nodes_time=ts.nodes_time,
        edges_left=ts.edges_left,
        edges_right=ts.edges_right,
        edges_parent=ts.edges_parent,
        edges_child=ts.edges_child,
        edge_insertion_order=ts.indexes_edge_insertion_order,
        edge_removal_order=ts.indexes_edge_removal_order,
        sequence_length=ts.sequence_length,
        **kwargs,
    )
    # sample genetic values
    g_samples = rv.run()
    # sample - individual assignment
    individuals = [i.id for i in ts.individuals()]
    samples_individuals = np.vstack([
        [n,k]
        for k, i in enumerate(individuals)
        for n in ts.individual(i).nodes])[:,1]
    # aggregate sample values to individuals
    g_individuals = np.apply_along_axis(
            bincount_fn, axis=0, arr=g_samples
            )
    
    return g_individuals

