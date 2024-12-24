import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import polars as pl

from core.currency import CurrencyPair
from preprocessing.transform_to_parquet import ParquetTransformer


def unzip_and_load_as_csv(zip_file_path: Path, temp_unzip_dir: Path) -> pd.DataFrame:
    # create second new temp_dir
    with zipfile.ZipFile(zip_file_path, mode="r") as zip_ref:
        csv_file: str = zip_ref.namelist()[0]
        zip_ref.extract(member=csv_file, path=temp_unzip_dir)  # unzip into temp_dir

        csv_file_path: Path = temp_unzip_dir.joinpath(csv_file)
        df_pandas: pd.DataFrame = pd.read_csv(csv_file_path, header=None)

        return df_pandas


def test_preprocess_zip_file() -> None:
    """
    This test makes sure that the way HiveDataset is created matches the result produced by simply unpacking csv file
    and reading it with pandas, we simply compare shapes of two dataframes
    """
    zipped_data_dir: Path = Path("D:/data/zipped_data")
    currency_pair: CurrencyPair = CurrencyPair(base="ADA", term="USDT")
    zip_file_name: str = "ADAUSDT-trades-2021-08.zip"

    with (
        tempfile.TemporaryDirectory(prefix="temp_hive_dir") as temp_hive_dir,
        tempfile.TemporaryDirectory(prefix="temp_unzip_dir") as temp_unzip_dir,
    ):
        temp_hive_dir: Path = Path(temp_hive_dir)
        temp_unzip_dir: Path = Path(temp_unzip_dir)

        transformer: ParquetTransformer = ParquetTransformer(
            zipped_data_dir=zipped_data_dir, temp_dir=temp_unzip_dir, output_dir=temp_hive_dir
        )
        transformer.preprocess_zip_file(currency_pair=currency_pair, zip_file=zip_file_name)
        # Collect number of rows lazily
        df_hive: pl.LazyFrame = pl.scan_parquet(source=temp_hive_dir)
        hive_shape: int = df_hive.collect().shape[0]

        # unzip data using pandas by simply unpacking csv file and then reading csv with pandas
        zip_file_path: Path = (
            zipped_data_dir
            .joinpath(currency_pair.name)
            .joinpath(zip_file_name)
        )

        df_pandas: pd.DataFrame = unzip_and_load_as_csv(
            zip_file_path=zip_file_path, temp_unzip_dir=temp_unzip_dir
        )

    assert df_pandas.shape[0] == hive_shape, "Polars HiveDataset and Pandas.DataFrame have different shapes"
