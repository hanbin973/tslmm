import numpy as np
import tskit
from numba import i4, f8
from numba.experimental import jitclass

# --- Generalized Trait ---

spec_generalized = [
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
        ('NULL', i4),
        ('cut_time', f8),
        ('lineages', i4[:]),
        ('mode_covariance', i4),
]

@jitclass(spec_generalized)
class GeneralizedTrait:
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
        sequence_length,
        cut_time,
        mode_covariance=0,
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
        
        
        self.cut_time = cut_time
        self.lineages = np.zeros(N, dtype=np.int32)
        self.mode_covariance = mode_covariance

        for j, u in enumerate(samples):
            self.num_samples[u] = 1
            # Add branch to the virtual sample
            v = num_nodes + 1 + j
            self.parent[v] = u
            self.num_samples[v] = 1
            
            # Initialize lineages for samples if they are below cut_time
            if self.nodes_time[u] <= self.cut_time:
                self.lineages[u] = 1

    def remove_edge(self, p, c):
        self.stack[c] += self.get_z(c)
        self.x[c] = self.position
        self.parent[c] = -1
        
        delta_lineages = 0
        if self.nodes_time[c] <= self.cut_time:
            delta_lineages = 1
        else:
            delta_lineages = self.lineages[c]
        
        # adjust path
        c_samples = self.num_samples[c]
        self.adjust_path_up(c, p, -1, delta_lineages, -c_samples)

    def insert_edge(self, p, c):
        delta_lineages = 0
        if self.nodes_time[c] <= self.cut_time:
            delta_lineages = 1
        else:
            delta_lineages = self.lineages[c]

        # Update path
        c_samples = self.num_samples[c]
        
        # Adjust up
        self.adjust_path_up(c, p, +1, delta_lineages, c_samples)
        self.x[c] = self.position
        self.parent[c] = p

    def adjust_path_up(self, c, p, sign, delta_lineages, delta_samples):
        # sign = -1 for removing edges, +1 for adding
        curr = p
        while curr != self.NULL:
            # We perform updates on 'curr'
            
            # 1. Stack (genetic value)
            self.stack[curr] += self.get_z(curr) 
            
            self.x[curr] = self.position
            # check for floating point error
            self.stack[c] -= sign * self.stack[curr]
            
            # Lineage update
            # lineages[p] depends on children.
            if self.nodes_time[curr] > self.cut_time:
                self.lineages[curr] += sign * delta_lineages

            # Sample update
            self.num_samples[curr] += delta_samples
            
            if self.parent[curr] == self.NULL:
                return curr
            curr = self.parent[curr]
        return -1 # Should not happen if p != NULL

    def get_z(self, u):
        p = self.parent[u]
        if p == self.NULL or u >= self.virtual_root:
            return 0.0
        
        upper = self.nodes_time[p]
        lower = self.nodes_time[u]
        
        # If the entire branch is below cut_time, contribution is 0
        if upper <= self.cut_time:
            return 0.0
            
        # If u is below cut_time, effective lower bound is cut_time
        eff_lower = lower
        if eff_lower < self.cut_time:
            eff_lower = self.cut_time
            
        time = upper - eff_lower
        
        span = self.position - self.x[u]
        
        if self.mode_covariance == 1:
            # Deterministic covariance calculation
            # Use lineages as weight
            weight = 0.0
            if self.nodes_time[u] > self.cut_time:
                 weight = float(self.lineages[u])
            else:
                 weight = 1.0 
            denom = 1.0 
            
            return time * span * weight / denom
        else:
            # Simulation
            return np.sqrt(time * span) * np.random.normal()

    def run(self):
        sequence_length = self.sequence_length
        m = self.edges_left.shape[0]
        
        M = self.edges_left.shape[0]
        in_order = self.edge_insertion_order
        out_order = self.edge_removal_order
        edges_left = self.edges_left
        edges_right = self.edges_right
        edges_parent = self.edges_parent
        edges_child = self.edges_child

        j = 0
        k = 0
        left = 0.0
        self.position = left

        while k < M and left < self.sequence_length:
            
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
            
            # Determine next event
            right = self.sequence_length
            if j < M:
                right = min(right, edges_left[in_order[j]])
            if k < M:
                right = min(right, edges_right[out_order[k]])
            
            left = right
            self.position = left
            
        # clear remaining things down to virtual samples
        for out_j, u in enumerate(self.samples):
            v = self.virtual_root + 1 + out_j
            self.remove_edge(u, v)

        out = np.zeros(len(self.samples))
        for out_i in range(len(self.samples)):
             i = out_i + self.virtual_root + 1
             out[out_i] = self.stack[i]
        return out


def sim_genetic_value_generalized(ts, cut_time, mode_covariance=0, **kwargs):
    def bincount_fn(w):
        return np.bincount(samples_individuals, w)
    
    rv = GeneralizedTrait(
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
        cut_time=cut_time,
        mode_covariance=mode_covariance,
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


# --- Active Lineage Counter ---

spec_active_lineage = [
    ('edges_left', f8[:]),
    ('edges_right', f8[:]),
    ('edges_parent', i4[:]),
    ('edges_child', i4[:]),
    ('edge_insertion_order', i4[:]),
    ('edge_removal_order', i4[:]),
    ('sequence_length', f8),
    ('nodes_time', f8[:]),
    ('cut_time', f8),
    ('total_lineages', i4),
    ('counts', i4[:]),
]

@jitclass(spec_active_lineage)
class ActiveLineageCounter:
    def __init__(
        self,
        nodes_time,
        edges_left,
        edges_right,
        edges_parent,
        edges_child,
        edge_insertion_order,
        edge_removal_order,
        sequence_length,
        cut_time,
        num_trees,
    ):
        self.edges_left = edges_left
        self.edges_right = edges_right
        self.edges_parent = edges_parent
        self.edges_child = edges_child
        self.edge_insertion_order = edge_insertion_order
        self.edge_removal_order = edge_removal_order
        self.sequence_length = sequence_length
        self.nodes_time = nodes_time
        self.cut_time = cut_time
        self.total_lineages = 0
        self.counts = np.zeros(num_trees, dtype=np.int32)

    def remove_edge(self, p, c):
        if self.nodes_time[c] <= self.cut_time < self.nodes_time[p]:
            self.total_lineages -= 1

    def insert_edge(self, p, c):
        if self.nodes_time[c] <= self.cut_time < self.nodes_time[p]:
            self.total_lineages += 1

    def run(self):
        M = self.edges_left.shape[0]
        in_order = self.edge_insertion_order
        out_order = self.edge_removal_order
        edges_left = self.edges_left
        edges_right = self.edges_right
        edges_parent = self.edges_parent
        edges_child = self.edges_child

        j = 0
        k = 0
        left = 0.0
        tree_idx = 0
        
        while k < M and left < self.sequence_length:
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
            
            right = self.sequence_length
            if j < M:
                right = min(right, edges_left[in_order[j]])
            if k < M:
                right = min(right, edges_right[out_order[k]])
            
            if right > left:
                # Valid tree interval
                if tree_idx < len(self.counts):
                    self.counts[tree_idx] = self.total_lineages
                    tree_idx += 1
            
            left = right
            
        return self.counts

def count_active_lineages(ts, cut_time):
    counter = ActiveLineageCounter(
        nodes_time=ts.nodes_time,
        edges_left=ts.edges_left,
        edges_right=ts.edges_right,
        edges_parent=ts.edges_parent,
        edges_child=ts.edges_child,
        edge_insertion_order=ts.indexes_edge_insertion_order,
        edge_removal_order=ts.indexes_edge_removal_order,
        sequence_length=ts.sequence_length,
        cut_time=cut_time,
        num_trees=ts.num_trees,
    )
    return counter.run()

# --- Generalized Accumulator ---

spec_accumulator = [
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
    ('NULL', i4),
    ('cut_time', f8),
    ('lineages', i4[:]),
    ('total_lineages', i4),
    ('mode_covariance', i4),
]

@jitclass(spec_accumulator)
class GeneralizedAccumulator:
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
        sequence_length,
        cut_time,
        mode_covariance=0,
    ):
        self.sample_weights = np.zeros(num_nodes, dtype=np.float64)
        self.parent = np.full(num_nodes + len(samples) + 1, -1, dtype=np.int32)
        self.num_samples = np.zeros(num_nodes + len(samples) + 1, dtype=np.int32)
        
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
        N = num_nodes + len(samples) + 1
        self.x = np.zeros(N, dtype=np.float64)
        self.w = np.zeros(N, dtype=np.float64)
        self.stack = np.zeros(N, dtype=np.float64)
        self.NULL = -1 # to avoid tskit.NULL in numba
        
        
        self.cut_time = cut_time
        self.lineages = np.zeros(N, dtype=np.int32)
        self.total_lineages = 0
        self.mode_covariance = mode_covariance

        for j, u in enumerate(samples):
            self.num_samples[u] = 1
            # Add branch to the virtual sample
            v = num_nodes + 1 + j
            self.parent[v] = u
            self.num_samples[v] = 1
            
            # Initialize lineages for samples if they are below cut_time
            if self.nodes_time[u] <= self.cut_time:
                self.lineages[u] = 1

    def remove_edge(self, p, c):
        self.stack[c] += self.get_z(c)
        self.x[c] = self.position
        self.parent[c] = -1
        
        delta_lineages = 0
        if self.nodes_time[c] <= self.cut_time:
            delta_lineages = 1
        else:
            delta_lineages = self.lineages[c]
        
        # adjust path
        c_samples = self.num_samples[c]
        self.adjust_path_up(c, p, -1, delta_lineages, -c_samples)
        
        # Update total_lineages based on ActiveLineageCounter logic
        if self.nodes_time[c] <= self.cut_time < self.nodes_time[p]:
            self.total_lineages -= 1

    def insert_edge(self, p, c):
        delta_lineages = 0
        if self.nodes_time[c] <= self.cut_time:
            delta_lineages = 1
        else:
            delta_lineages = self.lineages[c]

        # Update path
        c_samples = self.num_samples[c]
        
        # Adjust up
        self.adjust_path_up(c, p, +1, delta_lineages, c_samples)
        self.x[c] = self.position
        self.parent[c] = p
        
        # Update total_lineages based on ActiveLineageCounter logic
        if self.nodes_time[c] <= self.cut_time < self.nodes_time[p]:
            self.total_lineages += 1

    def adjust_path_up(self, c, p, sign, delta_lineages, delta_samples):
        # sign = -1 for removing edges, +1 for adding
        curr = p
        while curr != self.NULL:
            # We perform updates on 'curr'
            
            # 1. Stack (genetic value)
            self.stack[curr] += self.get_z(curr) 
            
            self.x[curr] = self.position
            # check for floating point error
            self.stack[c] -= sign * self.stack[curr]
            
            # Lineage update
            # lineages[p] depends on children.
            if self.nodes_time[curr] > self.cut_time:
                self.lineages[curr] += sign * delta_lineages

            # Sample update
            self.num_samples[curr] += delta_samples
            
            if self.parent[curr] == self.NULL:
                return curr
            curr = self.parent[curr]
        return -1 # Should not happen if p != NULL

    def get_z(self, u):
        p = self.parent[u]
        if p == self.NULL or u >= self.virtual_root:
            return 0.0
        
        upper = self.nodes_time[p]
        lower = self.nodes_time[u]
        
        # If the entire branch is below cut_time, contribution is 0
        if upper <= self.cut_time:
            return 0.0
            
        # If u is below cut_time, effective lower bound is cut_time
        eff_lower = lower
        if eff_lower < self.cut_time:
            eff_lower = self.cut_time
            
        time = upper - eff_lower
        
        span = self.position - self.x[u]
        
        if self.mode_covariance == 1:
            # Deterministic covariance calculation
            # Use lineages as weight
            weight = 0.0
            if self.nodes_time[u] > self.cut_time:
                 weight = float(self.lineages[u])
            else:
                 weight = 1.0 
            
            # Normalize by total_lineages
            denom = 1.0
            if self.total_lineages > 0:
                denom = float(self.total_lineages)
            
            return time * span * weight / denom
        
        return 0.0

    def run(self):
        M = self.edges_left.shape[0]
        in_order = self.edge_insertion_order
        out_order = self.edge_removal_order
        edges_left = self.edges_left
        edges_right = self.edges_right
        edges_parent = self.edges_parent
        edges_child = self.edges_child

        j = 0
        k = 0
        left = 0.0
        self.position = left

        while k < M and left < self.sequence_length:
            
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
            
            # Determine next event
            right = self.sequence_length
            if j < M:
                right = min(right, edges_left[in_order[j]])
            if k < M:
                right = min(right, edges_right[out_order[k]])
            
            left = right
            self.position = left
            
        # clear remaining things down to virtual samples
        for out_j, u in enumerate(self.samples):
            v = self.virtual_root + 1 + out_j
            self.remove_edge(u, v)

        out = np.zeros((len(self.samples)))
        for out_i in range(len(self.samples)):
             i = out_i + self.virtual_root + 1
             out[out_i] = self.stack[i]
        return out


def sim_normalized_covariance(ts, cut_time, mode_covariance=1, **kwargs):
    def bincount_fn(w):
        return np.bincount(samples_individuals, w)
    
    rv = GeneralizedAccumulator(
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
        cut_time=cut_time,
        mode_covariance=mode_covariance,
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
