import sys
sys.path.append(r'C:\Users\310\Desktop\Progects_Py\microstructure_price_prediction\src')

from datetime import datetime

import polars as pl

from core.currency import CurrencyPair


def main():
    df: pl.LazyFrame = pl.scan_parquet(r"C:\Users\310\Desktop\Progects_Py\data\microstructure_price_prediction_data\unzipped")
    currency_pair: CurrencyPair = CurrencyPair(base="DOGE", term="USDT")

    df = df.filter(
        (pl.col("date") >= datetime(2024, 6, 1)) &
        (pl.col("date") < datetime(2024, 7, 31))
    )

    print(df.collect())
    


if __name__ == "__main__":
    main()
