from datetime import datetime, timedelta
import time
import polars as pl

def time_to_end(start_of_period: datetime,
                end_of_period: datetime,  
                delta: timedelta, 
                current_end_bound: datetime,
                elapsed_time: time) -> None:
    
    total_iterations = (end_of_period - start_of_period) // delta
    current_iterarion = (current_end_bound - start_of_period) / delta
    remaining_iterations = total_iterations - current_iterarion 
    fraction_completed = current_iterarion / total_iterations

    # Average time of execution of the loop
    average_time = elapsed_time / current_iterarion

    time_remaning = average_time * remaining_iterations

    print(f"Time remaining: {time_remaning},\nFraction completed: {fraction_completed}")


def insert_crossection_id(cross_section_id: datetime, 
                          df_cross_section: pl.DataFrame) -> pl.DataFrame:
    
    id_series = pl.Series("cross_section_id" , [cross_section_id] * df_cross_section.shape[0])
    return df_cross_section.insert_column(1, id_series)

    

