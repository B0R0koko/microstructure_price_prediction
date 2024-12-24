from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from functools import partial
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import List, Dict, Any

import polars as pl
from tqdm import tqdm

from core.currency import CurrencyPair
from core.time_utils import Bounds, TimeOffset


class FeaturePipeline(ABC):

    def __init__(self, hive_dir: Path):
        self.hive_dir: Path = hive_dir
        self.return_timedelta: TimeOffset = TimeOffset.MINUTE

    def _get_currency_pairs_cross_section(self, bounds: Bounds) -> List[CurrencyPair]:
        """
        Returns a list of CurrencyPair for which there is data stored in self.hive_dir: Path for the
        given time interval
        """
        df_hive: pl.LazyFrame = pl.scan_parquet(self.hive_dir, hive_partitioning=True)
        # Extract dates of start_time and end_time

        unique_symbols: List[str] = (
            df_hive
            .filter(
                pl.col("date").is_between(lower_bound=bounds.start_inclusive.date(),
                                          upper_bound=bounds.end_exclusive.date()) &
                pl.col("trade_time").is_between(lower_bound=bounds.start_inclusive, upper_bound=bounds.end_exclusive)
            )
            .select("symbol").unique().collect()["symbol"].to_list()
        )

        return [CurrencyPair.from_string(symbol=symbol) for symbol in unique_symbols]

    def load_currency_pair_dataframe(self, currency_pair: CurrencyPair, bounds: Bounds) -> pl.DataFrame:
        """Load data for a given CurrencyPair with specific time interval [start_time, end_time + return_timedelta)"""

        df_hive: pl.LazyFrame = pl.scan_parquet(self.hive_dir, hive_partitioning=True)
        effective_end_time: datetime = bounds.end_exclusive + self.return_timedelta.value
        df_currency_pair: pl.LazyFrame = df_hive.filter(
            (pl.col("symbol") == currency_pair.name) &
            # Load data by filtering by both hive folder structure and columns inside each parquet file
            (pl.col("date").is_between(lower_bound=bounds.start_inclusive.date(),
                                       upper_bound=bounds.end_exclusive.date())) &
            (pl.col("trade_time").is_between(lower_bound=bounds.start_inclusive,
                                             upper_bound=effective_end_time))
        )

        return df_currency_pair.collect()

    def attach_currency_pair_return(self, df_currency_pair: pl.DataFrame, bounds: Bounds) -> float:
        """
        Attaches return column for a specified timedelta, how far into the future we would like to compute the return.
        Returns bounds.end + time_offset return for the specified CurrencyPair
        """
        effective_end_time: datetime = bounds.end_exclusive + self.return_timedelta.value
        currency_pair_log_return: float = (
            df_currency_pair
            .filter(
                pl.col("trade_time").is_between(lower_bound=bounds.end_exclusive, upper_bound=effective_end_time)
            )
            .select((pl.col("price").last() / pl.col("price").first()).log())
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
        cross_section_features: List[Dict[str, Any]] = []

        for currency_pair in currency_pairs:
            # Load and collect pl.DataFrame for current CurrencyPair, read to RAM no avoid calling collect multiple times
            df_currency_pair: pl.DataFrame = self.load_currency_pair_dataframe(
                currency_pair=currency_pair, bounds=bounds
            )
            # Compute features using loaded pl.DataFrame
            currency_pair_features: Dict[str, Any] = self.compute_features_for_currency_pair(
                currency_pair=currency_pair, df_currency_pair=df_currency_pair, bounds=bounds
            )
            currency_pair_features["log_return"] = self.attach_currency_pair_return(
                df_currency_pair=df_currency_pair, bounds=bounds
            )
            cross_section_features.append(currency_pair_features)

        df_cross_section: pl.DataFrame = pl.DataFrame(cross_section_features)
        df_cross_section = df_cross_section.with_columns(
            pl.lit(value=bounds.start_inclusive).alias("cross_section_start_time"),
            pl.lit(value=bounds.end_exclusive).alias("cross_section_end_time"),
        )
        return df_cross_section

    # Parallelize this function to be able to run at least using 10 processes
    def load_multiple_cross_sections(self, cross_section_bounds: List[Bounds]) -> pl.DataFrame:
        dfs: List[pl.DataFrame] = []

        with (
            tqdm(total=len(cross_section_bounds), desc="Computing cross-sections with multiprocessing: ") as pbar,
            Pool(processes=10) as pool,
        ):
            promises: List[AsyncResult] = []

            for bounds in cross_section_bounds:
                promise: AsyncResult = pool.apply_async(
                    partial(self.load_cross_section, bounds=bounds)
                )
                promises.append(promise)

            for promise in promises:
                df_cross_section: pl.DataFrame = promise.get()  # fetch output of self.load_cross_section from Future
                dfs.append(df_cross_section)

                pbar.update(1)

        return pl.concat(dfs)

    @abstractmethod
    def compute_features_for_currency_pair(
            self, currency_pair: CurrencyPair, df_currency_pair: pl.DataFrame, bounds: Bounds
    ) -> Dict[str, Any]:
        """
        Define logic of how we load data for a passed in currency_pair as well as how we compute features. This method
        must return pl.DataFrame with all compute features. Later this dataframe will be attached to df_cross_sections:
        pl.DataFrame
        """


def _test_main() -> None:
    hive_dir: Path = Path("D:/data/transformed_data")
    start_time: datetime = datetime(2024, 11, 1, 0, 0, 0)
    end_time: datetime = datetime(2024, 11, 1, 0, 0)
    step: timedelta = timedelta(seconds=5)
    interval: timedelta = timedelta(minutes=15)

    bounds: Bounds = Bounds(start_inclusive=start_time, end_exclusive=end_time)
    cross_section_bounds: List[Bounds] = bounds.generate_overlapping_bounds(step=step, interval=interval)

    pipeline: FeaturePipeline = FeaturePipeline(hive_dir=hive_dir)
    pipeline.load_multiple_cross_sections(cross_section_bounds=cross_section_bounds)


if __name__ == "__main__":
    _test_main()
