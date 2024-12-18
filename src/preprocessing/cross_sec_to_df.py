import sys
sys.path.append(r"C:\Users\310\Desktop\Progects_Py\microstructure_price_prediction\src")

from datetime import datetime, timedelta
from pathlib import Path
import polars as pl
import time

from core.time_utils import Bounds
from preprocessing.microstructure_pipeline import MicrostructurePipeline
from stacking_utils import time_to_end, insert_crossection_id

def main():
    """
    Args:
        hive_dir: Path - directory from data is taken
        save_dir: Path - directory where dataframe will be saved after execution
        save: Bool - flag, weather you want to save file or not
        start_of period: datetime - the begining of time period concidered for dataframe creation
        end_of_period: datetime - the end ot time period concidered for dataframe creation
        delta: time_delta - the length of a window that used to calculate one cross-section
    Overview:
        Overall this function calls MicrostructurePipeline and calculate crossections. 
        After cross-section is calculeted it is stacked with another ones. 
        When all cross-sections created and stacked the function can save the resulting df in spesified directory.
    """


    hive_dir: Path = Path(r"C:\Users\310\Desktop\Progects_Py\data\microstructure_price_prediction_data\unzipped")
    save_dir: Path = Path(r"C:\Users\310\Desktop\Progects_Py\data\microstructure_price_prediction_data\dfs")
    save: bool = True
    start_of_period: datetime = datetime(2024, 6, 15,)
    end_of_period: datetime = datetime(2024, 7, 15,)
    delta: timedelta = timedelta(seconds=10 )


    bounds: Bounds = Bounds(start_inclusive=start_of_period, end_exclusive=start_of_period + delta)
    staked_df = pl.DataFrame()

    # Record the start time
    start_time = time.time()

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
        df_cross_section = insert_crossection_id(cross_section_id=cross_section_id, df_cross_section=df_cross_section)

        # Stack dfs together
        staked_df = staked_df.vstack(df_cross_section) if not staked_df.is_empty() else df_cross_section

        bounds.start_inclusive += delta
        bounds.end_exclusive += delta

        # Keep track of how much time have passed
        elapsed_time = time.time() - start_time
        time_to_end(start_of_period=start_of_period,
                    end_of_period=end_of_period,
                    delta=delta,
                    current_end_bound=bounds.end_exclusive,
                    elapsed_time=elapsed_time)

    print(staked_df)

    if save:
        #Spesify file name 

        file_name = f"{start_of_period}_{end_of_period}_delta_{delta}_return_5_sec.parquet"

        file_path = save_dir / file_name.replace(":", "-")
        staked_df.write_parquet(file_path)

        print(f"DataFrame saved to {file_path}")


if __name__ == "__main__":
    main()
