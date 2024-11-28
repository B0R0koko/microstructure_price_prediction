from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import polars as pl
from tqdm import tqdm

from core.currency import CurrencyPair
from core.time_utils import Bounds, TimeOffset


class FeaturePipeline(ABC):

    def __init__(self, hive_dir: Path):
        self.hive_dir: Path = hive_dir

    def _get_currency_pairs_cross_section(self, bounds: Bounds) -> List[CurrencyPair]:
        """
        Returns a list of CurrencyPair for which there is data stored in self.hive_dir: Path for the
        given time interval
        """
        df_hive: pl.LazyFrame = pl.scan_parquet(self.hive_dir)
        # Extract dates of start_time and end_time

        unique_symbols: List[str] = (
            df_hive.filter(
                pl.col("date").is_between(lower_bound=bounds.start_time.date(), upper_bound=bounds.end_time.date()),
            )
            .select("symbol").unique().collect()["symbol"].to_list()
        )

        return [CurrencyPair.from_string(symbol=symbol) for symbol in unique_symbols]

    def load_currency_pair_dataframe(self, currency_pair: CurrencyPair, bounds: Bounds) -> pl.LazyFrame:
        """Load data for a given CurrencyPair with specific time interval [start_time, end_time)"""

        df_hive: pl.LazyFrame = pl.scan_parquet(self.hive_dir)

        df_currency_pair: pl.LazyFrame = df_hive.filter(
            (pl.col("symbol") == currency_pair.name) &
            # Load data by filtering by both hive folder structure and columns inside each parquet file
            (pl.col("date").is_between(lower_bound=bounds.start_time.date(), upper_bound=bounds.end_time.date())) &
            (pl.col("trade_time").is_between(lower_bound=bounds.start_time, upper_bound=bounds.end_time))
        )

        return df_currency_pair

    def attach_currency_pair_return(
            self, currency_pair: CurrencyPair, bounds: Bounds, time_offset: TimeOffset
    ) -> float:
        """
        Attaches return column for a specified timedelta, how far into the future we would like to compute the return.
        Returns bounds.end + time_offset return for the specified CurrencyPair
        """
        df_hive: pl.LazyFrame = pl.scan_parquet(self.hive_dir)
        effective_end_time: datetime = bounds.end_time + time_offset.value  # find end boundary with datetime of return

        df_currency_pair_return: pl.LazyFrame = df_hive.filter(
            (pl.col("symbol") == currency_pair.name) &
            (pl.col("date").is_between(lower_bound=bounds.start_time.date(), upper_bound=effective_end_time.date())) &
            (pl.col("trade_time").is_between(lower_bound=bounds.end_time, upper_bound=effective_end_time))
        )
        currency_pair_log_return: float = (
            df_currency_pair_return
            .sort(by="trade_time", descending=False)
            .select(
                (pl.col("price").last() / pl.col("price").first()).log()
            )
            .collect()
            .item()
        )
        return currency_pair_log_return

    def load_cross_section(self, bounds: Bounds) -> pl.DataFrame:
        """
        This function runs self.compute_features_for_currency_pair for each of the currency_pair available within
        a given range of time defined by passed in start_time and end_time. Returns pl.DataFrame with all features
        attached
        """
        currency_pairs: List[CurrencyPair] = self._get_currency_pairs_cross_section(bounds=bounds)
        pbar = tqdm(currency_pairs)

        cross_section_features: List[Dict[str, Any]] = []

        for currency_pair in pbar:
            pbar.set_description(desc=f"Computing features for {currency_pair.name}")
            currency_pair_features: Dict[str, Any] = self.compute_features_for_currency_pair(
                currency_pair=currency_pair, bounds=bounds
            )
            currency_pair_features["log_return"] = self.attach_currency_pair_return(
                currency_pair=currency_pair, bounds=bounds, time_offset=TimeOffset.FIVE_SECONDS
            )
            cross_section_features.append(currency_pair_features)

        return pl.DataFrame(cross_section_features)

    @abstractmethod
    def compute_features_for_currency_pair(self, currency_pair: CurrencyPair, bounds: Bounds) -> Dict[str, Any]:
        """
        Define logic of how we load data for a passed in currency_pair as well as how we compute features. This method
        must return pl.DataFrame with all compute features. Later this dataframe will be attached to df_cross_sections:
        pl.DataFrame
        """


def _test_main() -> None:
    hive_dir: Path = Path("D:/data/transformed_data")
    start_time: datetime = datetime(2024, 9, 1)
    end_time: datetime = datetime(2024, 9, 20)

    bounds: Bounds = Bounds(start_time=start_time, end_time=end_time)

    pipeline: FeaturePipeline = FeaturePipeline(hive_dir=hive_dir)
    pipeline.load_cross_section(bounds=bounds)


if __name__ == "__main__":
    _test_main()
