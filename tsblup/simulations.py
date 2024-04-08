import numpy as np
import pandas as pd

import tskit
import tstrait
import msprime

class GeneticValueSimulator:
    def __init__(self, ts, beta):
        assert ts.sequence_length == beta.shape[0], "Length of beta should match the sequence length"
        self.ts = ts
        self.beta = beta

    def sim_mutations(self, **kwargs):
        self.mts = msprime.sim_mutations(self.ts, **kwargs)
        
    def sim_genetic_value(self, **kwargs):
        self.sim_mutations(**kwargs)
        trait_dict = {
            "position": self.mts.sites_position[self.mts.mutations_site],
            "site_id": self.mts.mutations_site,
            "edge_id": [m.edge for m in self.mts.mutations()],
            "effect_size": self.beta[self.mts.sites_position[self.mts.mutations_site].astype(int)],
            "trait_id": np.zeros(self.mts.num_mutations, dtype=np.int32),
            "causal_allele": [m.derived_state for m in self.mts.mutations()]
        }
        self.trait_df = pd.DataFrame(trait_dict)
        self.genetic_df = tstrait.genetic_value(self.mts, self.trait_df)
        
    def edge_effects(self):
        print("Not yet implemented")
        print("Workaround: check if the tree sequence is chunked")
