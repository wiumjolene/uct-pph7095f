import pandas as pd
import numpy as np
import config
import os
from scipy.stats import ttest_ind, chi2_contingency
# from statsmodels.stats.weightstats import DescrStatsW


class DataLoader:

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None

    def load_data(self):

        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)

            # --- 1. Set seed: 1987090901 ---
            np.random.seed(1987090901)

            # --- 2. Unique IDs ---
            unique_ids = self.df["id"].unique()

            # --- 3. Sample 70% ---
            sample_size = int(0.7 * len(unique_ids))
            sampled_ids = np.random.choice(unique_ids, size=sample_size, replace=False)

            # --- 4. Filter full longitudinal profiles ---
            self.df = self.df[self.df["id"].isin(sampled_ids)].copy()

            self.df = self.df[['id', 'obs_time', 'satisfaction', 'sex', 'age_marriage', 'cohab', 'income', 'hw_all']]

            return self.df

        else:
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        
