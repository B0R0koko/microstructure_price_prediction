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

    def compute_features_for_currency_pair(
            self, currency_pair: CurrencyPair, df_currency_pair: pl.DataFrame, bounds: Bounds
    ) -> Dict[str, Any]:
        """
        Given the data from df_currency_pair, call compute_features function on it
        which returns a mapping of feature names to their corresponding values
        """
        features: Dict[str, Any] = compute_features(
            df_currency_pair=df_currency_pair, currency_pair=currency_pair, bounds=bounds
        )
        return features


def _test_main():
    hive_dir: Path = Path("D:/data/transformed_data")
    start_time: datetime = datetime(2024, 11, 1, 0, 0, 0)
    end_time: datetime = datetime(2024, 11, 3, 0, 0, 0)
    step: timedelta = timedelta(minutes=15)
    interval: timedelta = timedelta(minutes=15)

    bounds: Bounds = Bounds(start_inclusive=start_time, end_exclusive=end_time)
    cross_section_bounds: List[Bounds] = bounds.generate_overlapping_bounds(step=step, interval=interval)

    pipeline: MicrostructurePipeline = MicrostructurePipeline(hive_dir=hive_dir)
    df_features: pl.DataFrame = pipeline.load_multiple_cross_sections(cross_section_bounds=cross_section_bounds)
    df_features.to_pandas().to_csv("D:/data/final/features_final_1.csv", index=False)


if __name__ == "__main__":
    _test_main()
