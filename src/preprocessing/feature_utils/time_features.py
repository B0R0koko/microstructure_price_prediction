import polars as pl
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


# Function to plot circular features
def plot_circular_time_features(df, time_units):
    """
    Plot circular features for sine and cosine values of each time unit.
    
    Parameters:
        df (pl.DataFrame): Polars DataFrame with sine and cosine features.
        time_units (dict): Dictionary of time units and their periods.
    """
    df_pandas = df.to_pandas()
    
    for unit in time_units.keys():
        sin_col = f"{unit}_sin"
        cos_col = f"{unit}_cos"
        
        if sin_col in df_pandas.columns and cos_col in df_pandas.columns:
            plt.figure(figsize=(6, 6))
            plt.scatter(df_pandas[sin_col], df_pandas[cos_col], alpha=0.7, s=10)
            plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
            plt.title(f"Circular Representation of {unit.capitalize()} Features")
            plt.xlabel(f"{unit}_sin")
            plt.ylabel(f"{unit}_cos")
            plt.gca().set_aspect('equal', adjustable='box')  # Ensure the plot is circular
            plt.grid()
            plt.show()


# Define a function to compute sine and cosine features for a time unit
def add_sin_cos_features(df, time_col, time_units, plot=False):
    """
    Add sine and cosine features for a time unit.
    
    Parameters:
        df (pl.DataFrame): Polars DataFrame.
        time_col (str): Name of the datetime column.
        time_units (dict): Dictionary of time units and their periods.
        plot (bool): Bool value that difine weather to call plot_circular_time_features() or not.
    
    Returns:
        pl.DataFrame: Polars DataFrame with added features.
    """
    for unit, period in time_units.items():
        # Extract the desired unit (e.g., minute, hour, etc.)
        df = df.with_columns(
            getattr(pl.col(time_col).dt, unit)().alias(f"{unit}")
        )

        # Map the unit value to radians [0, 2π]
        df = df.with_columns(
            (2 * np.pi * pl.col(f"{unit}") / period).alias(f"{unit}_radians")
        )

        # Add sine and cosine features
        df = df.with_columns(
            [
                pl.col(f"{unit}_radians").sin().alias(f"{unit}_sin"),
                pl.col(f"{unit}_radians").cos().alias(f"{unit}_cos")
            ]
        )

        # Drop intermediate radians column and units
        df = df.drop(f"{unit}_radians", f"{unit}")

    if plot:
        plot_circular_time_features(df, time_units)
    
    return df




# TESTING ###############################################
start_datetime = datetime(2024, 1, 1, 0, 0, 0, 0)  
end_datetime = datetime(2025, 1, 2, 0, 0, 1, 0)    

step_minutes = 1  

# Generate the list of datetime objects
datetime_list = []
current_datetime = start_datetime
while current_datetime <= end_datetime:
    datetime_list.append(current_datetime)
    current_datetime += timedelta(minutes=step_minutes)

# Define time units
time_units = {
    "minute": 60,
    "hour": 24,
    "day": 7,
    "week": 52,
    "month": 12,
    "year": 365
}

# Create test pl.Df
df = pl.DataFrame({
    "trade_time": datetime_list
})
##########################################################

#df = add_sin_cos_features(df, "trade_time", time_units, False)






