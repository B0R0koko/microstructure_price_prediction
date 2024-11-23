import polars as pl

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