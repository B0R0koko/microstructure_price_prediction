import polars as pl
import numpy as np

class DataPrepare():

    def __init__(self, df: pl.DataFrame):
        """
        Initialize the DataPrepare class with a Polars DataFrame.
        """
        self.df = df
        self.df_train = None
        self.df_test = None

    def train_test_split(self, train_test_ratio: float = 0.8, print_stat: bool=True, cols_to_exclude: list[str]=None) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Splits the dataset into training and testing sets based on a train-test ratio.
        """
        if not isinstance(train_test_ratio, (float, int)) or not (0.0 < train_test_ratio < 1.0):
            raise ValueError("train_test_ratio must be a fraction between 0 and 1 (exclusive).")
        
        print(f'Train test ratio is {train_test_ratio}')

        df_train = pl.DataFrame()
        df_test = pl.DataFrame()

        df = self.df

        if not cols_to_exclude is None:
            df = df.select(pl.exclude(cols_to_exclude))

        for currency_pair in df["symbol"].unique():

            curr_df = df.filter(pl.col('symbol') == currency_pair)
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
            .group_by("symbol")
            .agg(pl.col(target_var).mean().alias("symbol_mean"))
        )

        df_train = self.df_train.join(symbol_means, on="symbol", how="left")

        # Step 3: Apply the same encoding to the test dataset to avoid leakeg
        df_test = self.df_test.join(symbol_means, on="symbol", how="left")

        return df_train.select(pl.exclude("symbol")), df_test.select(pl.exclude("symbol"))
    
    
    def X_y_split(self, target_var: str, target_encode: bool=False,) -> tuple[pl.DataFrame, np.ndarray, pl.DataFrame, np.ndarray]:
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

        df_train = df_train.drop_nulls()
        X_train = df_train.select(pl.exclude(target_var))
        y_train = df_train.select(pl.col(target_var)).to_numpy().ravel()

        df_test = df_test.drop_nulls()
        X_test = df_test.select(pl.exclude(target_var))
        y_test = df_test.select(pl.col(target_var)).to_numpy().ravel()

        return X_train, y_train, X_test, y_test


############################## SAME THING BUT IN SEPARATE FUNCTIONS ########################################


def train_test_split(df: pl.DataFrame, train_test_ratio = 0.5, print_stat=True) -> pl.DataFrame:

    df_train = pl.DataFrame()
    df_test = pl.DataFrame()

    for currency_pair in df["symbol"].unique():

        curr_df = df.filter(pl.col('symbol') == currency_pair)
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
        .group_by("symbol")
        .agg(pl.col(target_var).mean().alias("symbol_mean"))
    )

    df_train = df_train.join(symbol_means, on="symbol", how="left")

    # Step 3: Apply the same encoding to the test dataset to avoid leakeg
    df_test = df_test.join(symbol_means, on="symbol", how="left")

    return df_train.select(pl.exclude("symbol")), df_test.select(pl.exclude("symbol"))

def X_y_split(target_var: str, df_train: pl.DataFrame, df_test: pl.DataFrame) -> pl.DataFrame:
    df_train = df_train.drop_nulls()
    X_train = df_train.select(pl.exclude(target_var))
    y_train = df_train.select(pl.col(target_var)).to_numpy().ravel()

    df_test = df_test.drop_nulls()
    X_test = df_test.select(pl.exclude(target_var))
    y_test = df_test.select(pl.col(target_var)).to_numpy().ravel()

    return X_train, y_train, X_test, y_test