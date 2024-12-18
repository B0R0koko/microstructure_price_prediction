from datetime import datetime, timedelta
import time
import polars as pl

def time_to_end(start_of_period: datetime,
                end_of_period: datetime,  
                delta: timedelta, 
                current_end_bound: datetime,
                elapsed_time: time) -> None:
    
    """
    Estimate the time remaining and fraction completed in dd-hh-mm-ss format.
    
    Args:
        start_of_period (datetime): The start time of the process.
        end_of_period (datetime): The end time of the process.
        delta (timedelta): The time step for each iteration.
        current_end_bound (datetime): The current end bound of the process.
        elapsed_time (float): Total elapsed time since the process started (in seconds).
    """
    
    total_iterations = (end_of_period - start_of_period) // delta
    current_iterarion = (current_end_bound - start_of_period) / delta
    remaining_iterations = total_iterations - current_iterarion 
    fraction_completed = current_iterarion / total_iterations

    # Average time of execution of the loop
    average_time = elapsed_time / current_iterarion

    time_remaining = average_time * remaining_iterations

    # Convert time_remaining to dd-hh-mm-ss format
    days = int(time_remaining // 86400)
    hours = int((time_remaining % 86400) // 3600)
    minutes = int((time_remaining % 3600) // 60)
    seconds = int(time_remaining % 60)

    print(f"Time remaining| d{days:02d}-h{hours:02d}-m{minutes:02d}-s{seconds:02d}")
    print(f"Fraction completed: {fraction_completed:.5%}")


def insert_crossection_id(cross_section_id: datetime, 
                          df_cross_section: pl.DataFrame) -> pl.DataFrame:
    
    id_series = pl.Series("cross_section_id" , [cross_section_id] * df_cross_section.shape[0])
    return df_cross_section.insert_column(1, id_series)

    

