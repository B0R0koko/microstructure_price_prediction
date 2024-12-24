import os
import zipfile
from pathlib import Path
from typing import Optional, Sequence

import polars as pl
from pyarrow import RecordBatch, csv

from core.columns import *
from core.currency import CurrencyPair

_BINANCE_LEAVE_COLS: List[str] = [PRICE, QUANTITY, TRADE_TIME, IS_BUYER_MAKER]


def save_to_pyarrow_hive(
        csv_file_path: Path,
        output_dir: Path,
        new_columns: List[str],
        leave_cols: Optional[Sequence[str]],
        currency_pair: CurrencyPair
) -> None:
    """
    Preprocess and partition data by date using PyArrow hive partitioning, optimized for large datasets.
    Specify partition to be able to query parquet files using pl.scan_parquet with filter queries
    """
    # Define CSV read options
    MB: int = 1024 ** 2

    read_options: csv.ReadOptions = csv.ReadOptions(
        column_names=new_columns,
        block_size=MB * 128,  # batch_size in megabytes
        use_threads=None
    )
    # subset leave_cols from new_columns
    convert_options: csv.ConvertOptions = csv.ConvertOptions(include_columns=leave_cols)

    # Use a context manager to ensure the CSV reader is closed
    with csv.open_csv(csv_file_path, read_options=read_options, convert_options=convert_options) as csv_reader:
        batch: RecordBatch

        for batch in csv_reader:
            df: pl.DataFrame = pl.from_arrow(data=batch)
            # Attach SYMBOL columns as well as cast TRADE_TIME from milliseconds Timestamp to datetime
            df = df.with_columns(
                pl.lit(currency_pair.name).alias(SYMBOL),
                pl.col(TRADE_TIME).cast(pl.Datetime(time_unit="ms")).alias(TRADE_TIME)
            )
            # Create date column from TRADE_TIME
            df = df.with_columns(pl.col(TRADE_TIME).dt.date().alias("date"))
            # Write batch to parquet using pyarrow hive, this way we will be able to query necessary parquet files
            # with sql like queries from the local filesystem
            df.write_parquet(
                output_dir,
                use_pyarrow=True,
                pyarrow_options={
                    "partition_cols": ["date", SYMBOL],  # define set of filters here
                    "existing_data_behavior": "overwrite_or_ignore",
                }
            )


def check_if_zipped_file_is_csv(file_path: str) -> bool:
    """Checks if zipped file stored is a csv"""
    return file_path.endswith(".csv")


class ParquetTransformer:

    def __init__(self, zipped_data_dir: Path, temp_dir: Path, output_dir: Path):
        self.zipped_data_dir: Path = zipped_data_dir
        self.temp_dir: Path = temp_dir
        self.output_dir: Path = output_dir

        self.currency_pairs: List[CurrencyPair] = self._parse_collected_currency_pairs()

    def _parse_collected_currency_pairs(self) -> List[CurrencyPair]:
        """Folders are named in the following pattern: BASE-TERM"""
        currency_pairs: List[CurrencyPair] = []
        for folder in os.listdir(self.zipped_data_dir):
            base, term = folder.split("-")
            currency_pairs.append(CurrencyPair(base=base, term=term))
        return currency_pairs

    def preprocess_zip_file(self, currency_pair: CurrencyPair, zip_file: str) -> None:
        zip_path: Path = (
            self.zipped_data_dir
            .joinpath(currency_pair.name)
            .joinpath(zip_file)
        )
        # unzip data stored in the zipped archive
        with zipfile.ZipFile(zip_path, mode="r") as zip_ref:
            csv_file: str = zip_ref.namelist()[0]  # name of csv file once zip file is unzipped

            if not check_if_zipped_file_is_csv(file_path=csv_file):
                raise ValueError("Unzipped file is not a csv file")

            zip_ref.extract(member=csv_file, path=self.temp_dir)  # extract zipped csv file to self.temp_dir
            csv_file_path: Path = self.temp_dir.joinpath(csv_file)

            save_to_pyarrow_hive(
                csv_file_path=csv_file_path,
                output_dir=self.output_dir,
                new_columns=BINANCE_TRADE_COLS,
                leave_cols=_BINANCE_LEAVE_COLS,
                currency_pair=currency_pair
            )

    def preprocess_zip_folders(self) -> None:
        """
        Data should be organized like this, then we will iterate over each CurrencyPair
        D:/DATA/ZIPPED_DATA
        ├───ADA-USDT
        ├───BNB-BTC
        ├───BTC-USDT
        ├───ETH-BTC
        ├───LTC-BTC
        └───NEO-BTC
        """

        for currency_pair in self.currency_pairs:
            currency_pair: CurrencyPair
            currency_pair_folder: Path = self.zipped_data_dir.joinpath(currency_pair.name)
            # Iterate over each zip file store in currency_pair_folder
            for file in os.listdir(currency_pair_folder):
                self.preprocess_zip_file(
                    currency_pair=currency_pair,
                    zip_file=file
                )


if __name__ == "__main__":
    transformer: ParquetTransformer = ParquetTransformer(
        zipped_data_dir=Path("D:/data/zipped_data"),
        temp_dir=Path("D:/data/temp"),
        output_dir=Path("D:/data/transformed_data"),
    )

    transformer.preprocess_zip_folders()
