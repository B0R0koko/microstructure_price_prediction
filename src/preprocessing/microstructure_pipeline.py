from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

import polars as pl

from core.currency import CurrencyPair
from core.feature_pipeline import FeaturePipeline
from core.time_utils import Bounds
from preprocessing.features.features_27_11 import compute_features


class MicrostructurePipeline(FeaturePipeline):
    """Define first feature pipeline here. Make sure to implement all methods from abstract parent class"""

    def __init__(self, hive_dir: Path):
        super().__init__(hive_dir=hive_dir)

    def compute_features_for_currency_pair(self, currency_pair: CurrencyPair, bounds: Bounds) -> Dict[str, Any]:
        """
        Load data using load_currency_pair_dataframe from parent class and then call compute_features function on it
        which returns a mapping of feature names to their corresponding values
        """
        df_currency_pair: pl.LazyFrame = self.load_currency_pair_dataframe(currency_pair=currency_pair, bounds=bounds)
        # Compute features using pl.LazyFrame, make sure to call .collect() on pl.LazyFrame at the very end
        # this way it is more efficient
        features: Dict[str, Any] = compute_features(
            df_currency_pair=df_currency_pair, currency_pair=currency_pair, bounds=bounds
        )
        return features


def _test_main():
    hive_dir: Path = Path("D:/data/transformed_data")
    start_time: datetime = datetime(2024, 11, 1, 0, 0, 0)
    end_time: datetime = datetime(2024, 11, 1, 1, 0, 0)
    step: timedelta = timedelta(minutes=5)
    interval: timedelta = timedelta(minutes=15)

    bounds: Bounds = Bounds(start_inclusive=start_time, end_exclusive=end_time)
    cross_section_bounds: List[Bounds] = bounds.generate_overlapping_bounds(step=step, interval=interval)

    pipeline: MicrostructurePipeline = MicrostructurePipeline(hive_dir=hive_dir)
    # Run multiprocessing pipeline for multiple cross-sections
    df: pl.DataFrame = pipeline.load_multiple_cross_sections(
        cross_section_bounds=cross_section_bounds
    )
    df.to_pandas().to_csv("D:/data/final/features.csv", index=False)


if __name__ == "__main__":
    _test_main()
