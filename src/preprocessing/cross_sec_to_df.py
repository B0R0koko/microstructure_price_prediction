import sys
sys.path.append(r"C:\Users\310\Desktop\Progects_Py\microstructure_price_prediction\src")

from datetime import datetime, timedelta
from pathlib import Path
import polars as pl

from core.time_utils import Bounds
from preprocessing.microstructure_pipeline import MicrostructurePipeline

def main():
    hive_dir: Path = Path(r"C:\Users\310\Desktop\Progects_Py\data\microstructure_price_prediction_data\unzipped")
    save_dir: Path = Path(r"C:\Users\310\Desktop\Progects_Py\data\microstructure_price_prediction_data\dfs")
    save = True
    start_of_period: datetime = datetime(2024, 6, 15,)
    end_of_period: datetime = datetime(2024, 7, 15,)
    delta: timedelta = timedelta(seconds=10 )


    bounds: Bounds = Bounds(start_inclusive=start_of_period, end_exclusive=start_of_period + delta)
    staked_df = pl.DataFrame()

    while bounds.end_exclusive <= end_of_period:
        # Load data, compute features and get cross-setion
        pipeline: MicrostructurePipeline = MicrostructurePipeline(hive_dir=hive_dir)
        df_cross_section: pl.DataFrame = pipeline.load_cross_section(bounds=bounds)

        # Check if there were no trades at all during spesified time window
        if df_cross_section.is_empty():
            bounds.start_inclusive += delta
            bounds.end_exclusive += delta
            continue

        # Define cross-section ID with the end of cross-siction period
        cross_section_id = bounds.end_exclusive
        
        # Create a series of IDs and insert it in cross-section
        id_series = pl.Series("cross_section_id" , [cross_section_id] * df_cross_section.shape[0])
        df_cross_section.insert_column(1, id_series)
        
        bounds.start_inclusive += delta
        bounds.end_exclusive += delta

        staked_df = staked_df.vstack(df_cross_section) if not staked_df.is_empty() else df_cross_section

    print(staked_df)

    if save:
        #Spesify file name 
        file_name = f"{start_of_period}_{end_of_period}_delta_{delta}_return_5_sec.parquet"

        file_path = save_dir / file_name.replace(":", "-")
        staked_df.write_parquet(file_path)

        print(f"DataFrame saved to {file_path}")


if __name__ == "__main__":
    main()
