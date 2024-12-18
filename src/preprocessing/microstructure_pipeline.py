from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import polars as pl

from core.currency import CurrencyPair
from core.feature_pipeline import FeaturePipeline
from core.time_utils import Bounds
from preprocessing.features.features_27_11 import compute_features


class MicrostructurePipeline(FeaturePipeline):
    """Define first feature pipeline here. Make sure to implement all methods from abstract parent class"""

    def __init__(self, hive_dir: Path):
        super().__init__(
            hive_dir=hive_dir
        )

    def compute_features_for_currency_pair(self, currency_pair: CurrencyPair, bounds: Bounds) -> Dict[str, Any]:
        """
        Load data using load_currency_pair_dataframe from parent class and then call compute_features function on it
        which returns a mapping of feature names to their corresponding values
        """
        df_currency_pair: pl.LazyFrame = self.load_currency_pair_dataframe(currency_pair=currency_pair, bounds=bounds)

        # Check if during specified time window there are no trades withing currency pair
        no_trades = df_currency_pair.collect().is_empty()
        if no_trades:
            print(" No trades")
            return None
        
        # Compute features using pl.LazyFrame, make sure to call .collect() on pl.LazyFrame at the very end
        # this way it is more efficient
        features: Dict[str, Any] = compute_features(
            df_currency_pair=df_currency_pair, currency_pair=currency_pair, bounds=bounds
        )
        return features


def _test_main():
    hive_dir: Path = Path("D:/data/transformed_data")
    start_time: datetime = datetime(2021, 1, 1, 10, 0, 0)
    end_time: datetime = datetime(2021, 1, 1, 11, 0, 0)

    bounds: Bounds = Bounds(start_inclusive=start_time, end_exclusive=end_time)

    pipeline: MicrostructurePipeline = MicrostructurePipeline(hive_dir=hive_dir)
    df_cross_section: pl.DataFrame = pipeline.load_cross_section(bounds=bounds)
    print(df_cross_section)


if __name__ == "__main__":
    _test_main()
