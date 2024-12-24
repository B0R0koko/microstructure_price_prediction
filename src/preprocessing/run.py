from datetime import datetime

import polars as pl

from core.currency import CurrencyPair


def main():
    df: pl.LazyFrame = pl.scan_parquet("D:/data/transformed_data")
    currency_pair: CurrencyPair = CurrencyPair(base="ADA", term="USDT")
    start_time: datetime = datetime(2024, 9, 20, 10, 5, 10)
    end_time: datetime = datetime(2024, 9, 20, 10, 5, 30)

    df = df.filter(
        (pl.col("date").is_between(lower_bound=start_time.date(), upper_bound=end_time.date())) &
        (pl.col("trade_time").is_between(lower_bound=start_time, upper_bound=end_time))
    )

    print(df.sort(by="trade_time").head().collect())


if __name__ == "__main__":
    main()
