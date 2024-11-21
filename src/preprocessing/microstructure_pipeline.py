import sys
sys.path.append(r'C:\Users\310\Desktop\Progects_Py\microstructure_price_prediction\src')

from datetime import datetime
from pathlib import Path

import polars as pl

from core.currency import CurrencyPair
from core.feature_pipeline import FeaturePipeline


class MicrostructurePipeline(FeaturePipeline):
    """Define first feature pipeline here. Make sure to implement all methods from abstract parent class"""

    def __init__(self, hive_dir: Path):
        super().__init__(
            hive_dir=hive_dir
        )

    def compute_features_for_currency_pair(
            self, currency_pair: CurrencyPair, start_time: datetime, end_time: datetime
    ) -> pl.DataFrame:
        # get reference to the hive and then filter it by currency
        df_hive: pl.LazyFrame = pl.scan_parquet(self.hive_dir)
        df_currency_pair: pl.LazyFrame = df_hive.filter(
            (pl.col("symbol") == currency_pair.name) &
            (pl.col("date").is_between(lower_bound=start_time, upper_bound=end_time))
        )
        # Compute features using pl.LazyFrame, make sure to call .collect() on pl.LazyFrame at the very end
        # this way it is more efficient
        df_currency_pair = df_currency_pair.with_columns(
            (pl.col("price") * pl.col("quantity")).alias("quote"),
            (pl.when(pl.col("is_buyer_maker") == True).then(pl.col("price")).otherwise(None).fill_null(strategy="forward")).alias("last_ask"),
            (pl.when(pl.col("is_buyer_maker") == False).then(pl.col('price')).otherwise(None).fill_null(strategy='forward')).alias("last_bid"),
        )

        
        df_currency_pair = df_currency_pair.with_columns(
            ((pl.col("last_bid") + pl.col("last_ask")) / 2).alias("target")
        )
        # Implement feature computation
        return df_currency_pair.collect()

# As a possible target: "target" - average between last bid and ask 

# Perfectly we would like to predict when market orders are pushing eather bid or ask. 

def _test_main():
    hive_dir: Path = Path(r"C:\Users\310\Desktop\Progects_Py\data\microstructure_price_prediction_data\unzipped")
    start_date: datetime = datetime(2024, 6, 1)
    end_date: datetime = datetime(2024, 7, 31)

    pipeline: MicrostructurePipeline = MicrostructurePipeline(hive_dir=hive_dir)
    df_cross_section: pl.DataFrame = pipeline.load_cross_section(start_time=start_date, end_time=end_date)
    print(df_cross_section)


if __name__ == "__main__":
    _test_main()
