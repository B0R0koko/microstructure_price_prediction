import polars as pl
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

############################################################
# Data Preparation and Dataset Class
############################################################

class TimeSeriesDataset(Dataset):
    def __init__(self, 
                 df: pl.DataFrame, 
                 target_var: str = "log_return",
                 cat_features = ['currency_pair'],
                 seq_length: int = 10):
        """
        df should be the training or testing dataframe.
        It must contain:
          - "currency_pair"
          - "cross_section_id" (time dimension)
          - multiple feature columns including `target_var`
        
        The dataset will:
        1. Sort by cross_section_id (time dimension).
        2. Identify all unique currency pairs.
        3. Pivot data so we have a time series matrix: time x currency_pair x features.
        4. Create input sequences of length seq_length and predict next step target_var.
        
        The final output for each item is:
          X: shape (num_features, num_currency_pairs, seq_length)
          y: shape (num_currency_pairs) (for the next time step)
        """
        self.seq_length = seq_length
        self.target_var = target_var
        
        df = df.sort(by="cross_section_id")
        
        currency_pairs = df.select(pl.col("currency_pair").unique().sort()).to_series().to_list()
        self.currency_pairs = currency_pairs
    
        # Get unique cross_section_ids in sorted order
        cross_section_ids = df.select(pl.col("cross_section_id").unique().sort()).to_series().to_list()
        self.cross_section_ids = cross_section_ids

        # We want a structure like:
        # For each cross_section_id and currency_pair, we have a row of features.
        # Let's pivot: rows = cross_section_id, columns = currency_pair for each feature.

        all_cols = df.columns
        non_feature_cols = ["currency_pair", "cross_section_id"]
        feature_cols = [c for c in all_cols if c not in non_feature_cols]
        
        # We'll create a wide format: For each feature, pivot currency_pair
        # This results in multiple pivot operations. Another approach:
        # 1) group by cross_section_id and pivot currency_pair to create a wide df for each feature,
        #    then concatenate horizontally.

        pivoted_features = []
        for fcol in feature_cols:
            pivoted = (df.select(["cross_section_id", "currency_pair", fcol])
                         .pivot(index="cross_section_id", values=fcol, on="currency_pair")
                         .sort("cross_section_id"))
            
            # Now rename columns to ensure uniqueness
            # pivoted's columns are: ["cross_section_id", "DOGEUSDT", "AVAXUSDT", ...] for example.
            # We will rename currency pair columns to "<feature>_<currency_pair>"

            new_cols = []
            for col_name in pivoted.columns:
                if col_name not in ["cross_section_id"]:
                    new_cols.append(f"{fcol}_{col_name}")
                else:
                    new_cols.append(col_name)
            pivoted = pivoted.rename(dict(zip(pivoted.columns, new_cols)))
            pivoted_features.append(pivoted)

        wide_df = pivoted_features[0]
        for pdf in pivoted_features[1:]:
            wide_df = wide_df.join(pdf, on="cross_section_id", how="inner")
        
        wide_df = wide_df.fill_null(0)
        wide_df = wide_df.fill_nan(0)
        # wide_df now has columns: cross_section_id, and for each feature and currency pair combination: a column.
        # The columns now are something like:
        # cross_section_id, DOGEUSDT, AVAXUSDT (for first feature)
        # DOGEUSDT, AVAXUSDT (for second feature), ...
        # The order might be tricky; we must reorganize them properly.

        times = wide_df["cross_section_id"].to_numpy()
        wide_df = wide_df.drop("cross_section_id")
        
        # The resulting wide_df columns are in pattern:
        # For each feature, we have columns for each currency pair.
        # The order in which columns appear: It's grouped by feature pivot calls.
        # Actually, they are appended in order: first feature pivot gives N currency pair columns,
        # then next feature pivot gives N currency pair columns, etc.

        # We know how we appended them: in `feature_cols` order, each producing len(currency_pairs) columns.
        # So we can reshape the underlying numpy array accordingly.
        arr = wide_df.to_numpy()
        # arr shape: (num_time_steps, num_features * num_currency_pairs)
        
        num_time_steps = arr.shape[0]
        num_features = len(feature_cols)
        num_currency_pairs = len(currency_pairs)
        
        # Reshape arr to (num_time_steps, num_features, num_currency_pairs)
        # Currently: arr has shape: (num_time_steps, num_features * num_currency_pairs)
        arr = arr.reshape(num_time_steps, num_features, num_currency_pairs)
        
        # Actually we have (time_steps, features, currency_pairs).
        # For a single batch item we want (features, currency_pairs, seq_length),

        self.data_array = arr
        self.times = times
        self.num_features = num_features
        self.num_currency_pairs = num_currency_pairs

        # We need to produce sequences of length seq_length and predict the next step target_var.
        # First find the index of target_var in feature_cols
        self.target_idx = feature_cols.index(target_var)
        
        # The maximum starting index for a sequence is len- (seq_length+1) because we need seq_length for input and +1 for target.
        self.max_idx = num_time_steps - (seq_length + 1)

    def __len__(self):
        return self.max_idx + 1

    def __getitem__(self, idx):
        # Input indices: [idx: idx+seq_length]
        # Target index: idx+seq_length (one step ahead)
        input_slice = slice(idx, idx + self.seq_length)
        target_idx = idx + self.seq_length

        # Extract input sequence (time, features, currency_pairs)
        # self.data_array shape: (time, features, currency_pairs)
        seq_data = self.data_array[input_slice]  # shape: (seq_length, features, currency_pairs)
        # We want (features, currency_pairs, seq_length)
        seq_data = np.transpose(seq_data, (1,2,0))  # (features, currency_pairs, seq_length)

        # Target: log_return at the next step for each currency pair
        # target at target_idx along target feature dimension
        target = self.data_array[target_idx, self.target_idx, :]  # shape: (num_currency_pairs,)

        # Convert to torch tensors
        X = torch.tensor(seq_data, dtype=torch.float32)
        y = torch.tensor(target, dtype=torch.float32)

        return X, y
    
############################################################
# Separate utils
############################################################

def mape(preds, targets, eps=1e-6):
    # Mean Absolute Percentage Error
    # MAPE = mean(|(y - y_pred)| / |y|)
    # Adding eps to avoid division by zero if target is zero.
    return (torch.mean(torch.abs((targets - preds) / (targets + eps))) * 100).item()