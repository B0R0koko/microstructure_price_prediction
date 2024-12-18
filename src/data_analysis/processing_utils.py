import polars as pl
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class DataPrepare():

    def __init__(self, df: pl.DataFrame):
        """
        Initialize the DataPrepare class with a Polars DataFrame.
        """
        self.df = df
        self.df_train = None
        self.df_test = None

    def normalize(self, cross_section_id: str, exclude_columns: list[str] = None) -> pl.DataFrame:
        """
        Normalize the DataFrame by groups defined in `cross_section_id`.
        
        Parameters:
        - cross_section_id (str): Column used to define groups for normalization.
        - exclude_columns (list[str]): List of columns to exclude from normalization.
        
        Returns:
        - pl.DataFrame: The normalized DataFrame with columns replaced.
        """
        if exclude_columns is None:
            exclude_columns = []

        # Exclude specified columns and cross_section_id
        features = [col for col in self.df.columns if col not in [cross_section_id] + exclude_columns]

        # Calculate mean and std for each feature grouped by `cross_section_id`
        grouped = self.df.group_by(cross_section_id).agg([
            pl.col(column).mean().alias(f"{column}_mean") for column in features
        ] + [
            # Use a conditional expression to replace zero std with 1
            pl.when(pl.col(column).std() == 0)
            .then(1.0)
            .otherwise(pl.col(column).std())
            .alias(f"{column}_std") for column in features
        ])
        
        df_normalized = self.df.join(grouped, on=cross_section_id)

        normalized_columns = [
            ((pl.col(column) - pl.col(f"{column}_mean")) / pl.col(f"{column}_std")).alias(column)
            for column in features
        ]
        df_normalized = df_normalized.with_columns(normalized_columns)

        self.df = df_normalized.select(self.df.columns)

        return self.df

    def train_test_split(self, train_test_ratio: float = 0.8, print_stat: bool=True, exclude_columns: list[str]=None) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Splits the dataset into training and testing sets based on a train-test ratio.
        """
        if not isinstance(train_test_ratio, (float, int)) or not (0.0 < train_test_ratio < 1.0):
            raise ValueError("train_test_ratio must be a fraction between 0 and 1 (exclusive).")
        
        print(f'Train test ratio is {train_test_ratio}')

        df_train = pl.DataFrame()
        df_test = pl.DataFrame()

        df = self.df

        if not exclude_columns is None:
            df = df.select(pl.exclude(exclude_columns))

        for currency_pair in df["currency_pair"].unique():

            curr_df = df.filter(pl.col('currency_pair') == currency_pair)
            split_indx = int(len(curr_df) * train_test_ratio)

            train = curr_df[:split_indx]
            test = curr_df[split_indx:]

            if print_stat:
                print(f'Train len for {currency_pair} is {len(train)}')
                print(f'Test len for {currency_pair} is {len(test)}')

            df_train = df_train.vstack(train) if not df_train.is_empty() else train
            df_test = df_test.vstack(test) if not df_test.is_empty() else test

        self.df_train = df_train
        self.df_test = df_test

        return df_train, df_test
    
    
    def target_encoding(self, target_var: str,) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Encodes the target variable using mean encoding based on the 'symbol' column.
        """
        if self.df_train is None or self.df_test is None:
            raise ValueError("Train and test sets are not initialized. Call train_test_split first.")
            
        if target_var not in self.df_train.columns or target_var not in self.df_test.columns:
            raise ValueError(f"Target variable '{target_var}' not found in the dataset.")
        
        symbol_means = (
            self.df_train
            .group_by("currency_pair")
            .agg(pl.col(target_var).mean().alias("currency_pair_mean"))
        )

        df_train = self.df_train.join(symbol_means, on="currency_pair", how="left")

        # Step 3: Apply the same encoding to the test dataset to avoid leakeg
        df_test = self.df_test.join(symbol_means, on="currency_pair", how="left")

        return df_train.select(pl.exclude("currency_pair")), df_test.select(pl.exclude("currency_pair"))
    
    
    def X_y_split(self, target_var: str, target_encode: bool=False, to_pandas: bool=False) -> tuple[pl.DataFrame, np.ndarray, pl.DataFrame, np.ndarray]:
        """
        Splits the dataset into features (X) and target (y) for training and testing.
        """
        if self.df_train is None or self.df_test is None:
            raise ValueError("Train and test sets are not initialized. Call train_test_split first.")
        
        if target_encode:
            df_train, df_test = self.target_encoding(target_var,)
        else:
            df_train = self.df_train
            df_test = self.df_test

        if target_var not in df_train.columns or target_var not in df_test.columns:
            raise ValueError(f"Target variable '{target_var}' not found in the dataset.")

        df_train = df_train.fill_null(strategy="zero")
        X_train = df_train.select(pl.exclude(target_var))
        y_train = df_train.select(pl.col(target_var)).to_numpy().ravel()

        df_test = df_test.fill_null(strategy="zero")
        X_test = df_test.select(pl.exclude(target_var))
        y_test = df_test.select(pl.col(target_var)).to_numpy().ravel()
        
        if to_pandas == True:
            return X_train.to_pandas(), y_train, X_test.to_pandas(), y_test

        return X_train, y_train, X_test, y_test
    
    def visualize(self, start_time: datetime, end_time: datetime, variables_to_plot: list[str],) -> None: 
        """
        Visualize selected variabals againts trade time in chosen time range, all variables ploted on the same plot.
        """
        symbols: list[str] = self.df.select(pl.col('currency_pair')).unique()

        df_for_plot: pl.DataFrame = self.df.filter((pl.col('currency_pair') == symbols[0]) &
                                              (pl.col('cross_section_id').is_between(lower_bound=start_time, upper_bound=end_time)))

        # Plot all variables on the same plot
        plt.figure(figsize=(12, 8))

        for var in variables_to_plot:
            plt.plot(df_for_plot['cross_section_id'], df_for_plot[var], label=var)

        plt.xlabel('Cross-section id')
        plt.ylabel('Values')
        plt.title('Variables over Trade Time')
        plt.legend()
        plt.grid()
        plt.show()
    
    


############################## SAME THING BUT IN SEPARATE FUNCTIONS ########################################


def train_test_split(df: pl.DataFrame, train_test_ratio = 0.5, print_stat=True) -> pl.DataFrame:

    df_train = pl.DataFrame()
    df_test = pl.DataFrame()

    for currency_pair in df["currency_pair"].unique():

        curr_df = df.filter(pl.col('currency_pair') == currency_pair)
        split_indx = int(len(curr_df) * train_test_ratio)

        train = curr_df[:split_indx]
        test = curr_df[split_indx:]

        if print_stat:
            print(f'Train len for {currency_pair} is {len(train)}')
            print(f'Test len for {currency_pair} is {len(test)}')

        df_train = df_train.vstack(train) if not df_train.is_empty() else train
        df_test = df_test.vstack(test) if not df_test.is_empty() else test

    return df_train, df_test

def target_encoding(target_var: str, df_train: pl.DataFrame, df_test: pl.DataFrame) -> pl.DataFrame:
    symbol_means = (
        df_train
        .group_by("currency_pair")
        .agg(pl.col(target_var).mean().alias("currency_pair_mean"))
    )

    df_train = df_train.join(symbol_means, on="currency_pair", how="left")

    # Step 3: Apply the same encoding to the test dataset to avoid leakeg
    df_test = df_test.join(symbol_means, on="currency_pair", how="left")

    return df_train.select(pl.exclude("currency_pair")), df_test.select(pl.exclude("currency_pair"))

def X_y_split(target_var: str, df_train: pl.DataFrame, df_test: pl.DataFrame) -> pl.DataFrame:
    df_train = df_train.drop_nulls()
    X_train = df_train.select(pl.exclude(target_var))
    y_train = df_train.select(pl.col(target_var)).to_numpy().ravel()

    df_test = df_test.drop_nulls()
    X_test = df_test.select(pl.exclude(target_var))
    y_test = df_test.select(pl.col(target_var)).to_numpy().ravel()

    return X_train, y_train, X_test, y_test

def visualize(df: pl.DataFrame, start_time: datetime, end_time: datetime, variables_to_plot: list[str],) -> None: 

    symbols: list[str] = df.select(pl.col('currency_pair')).unique()

    df_for_plot: pl.DataFrame = df.filter((pl.col('currency_pair') == symbols[0]) &
                                          (pl.col('trade_time').is_between(lower_bound=start_time, upper_bound=end_time)))

    # Plot all variables on the same plot
    plt.figure(figsize=(12, 8))

    for var in variables_to_plot:
        plt.plot(df_for_plot['trade_time'], df_for_plot[var], label=var)

    plt.xlabel('Trade Time')
    plt.ylabel('Values')
    plt.title('Variables over Trade Time')
    plt.legend()
    plt.grid()
    plt.show()
