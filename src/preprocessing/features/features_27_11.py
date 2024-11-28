from typing import Dict, Any, List

import numpy as np
import polars as pl

from core.currency import CurrencyPair
from core.time_utils import TimeOffset, Bounds


def compute_slippage(df_trades: pl.LazyFrame) -> pl.LazyFrame:
    """
    Compute slippage as the difference between the actual amount of quote asset spent and the amount that could
    have been spent had all been executed at price_first
    """
    df_trades = df_trades.with_columns(
        quote_slippage_abs=(pl.col("quote_abs") - pl.col("price_first") * pl.col("quantity_abs")).abs()
    )
    df_trades = df_trades.with_columns(
        quote_slippage_sign=pl.col("quote_slippage_abs") * pl.col("quantity_sign").sign()
    )
    return df_trades


def aggregate_ticks_into_trades(df_currency_pair: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate ticks into trades using ns timestamps"""
    df_trades: pl.LazyFrame = (
        df_currency_pair
        .group_by("trade_time")
        .agg(
            price_first=pl.col("price").first(),  # if someone placed a trade with price impact, then price_first
            price_last=pl.col("price").last(),  # and price_last will differ
            # Amount spent in quote asset for the trade
            quote_abs=pl.col("quote_abs").sum(),
            quote_sign=pl.col("quote_sign").sum(),
            # Amount of base asset transacted
            quantity_abs=pl.col("quantity").sum(),
            quantity_sign=pl.col("quantity_sign").sum(),
        )
    )
    return df_trades


def compute_volume_imbalance(
        df_trades: pl.LazyFrame, bounds: Bounds, time_offsets: List[TimeOffset]
) -> Dict[str, float]:
    """Returns a dict of volume imbalance features computed using different offsets from end_date"""

    return {
        f"volume_imbalance_{offset.name}": (
            df_trades
            .filter(pl.col("trade_time") >= bounds.end_time - offset.value)
            .select(pl.col("quote_sign").sum() / pl.col("quote_abs").sum())
            .collect()
            .item()
        )
        for offset in time_offsets
    }


def compute_slippage_features(df_trades: pl.LazyFrame, bounds: Bounds, time_offsets: List[TimeOffset]) -> Dict[
    str, float]:
    """Compute slippage features based on quote_slippage_abs and quote_slippage_sign fields"""

    return {
        f"slippage_imbalance_{offset.name}": (
            df_trades
            .filter(pl.col("trade_time") >= bounds.end_time - offset.value)
            .select(pl.col("quote_slippage_sign").sum() / pl.col("quote_slippage_abs").sum())
            .collect()
            .item()
        )
        for offset in time_offsets
    }


def compute_features(df_currency_pair: pl.LazyFrame, currency_pair: CurrencyPair, bounds: Bounds) -> Dict[str, Any]:
    """Computes features for a give currency pair"""
    # Compute quote_volumes
    df_currency_pair = df_currency_pair.with_columns(
        quote_abs=(pl.col("price") * pl.col("quantity")),  # absolute value of quote transacted
        # When is_buyer_maker is True => someone came in and matched existing BID order => SELL
        # When is_buyer_maker is False => buyer came on and matched existing ASK order => BUY
        side_sign=1 - 2 * pl.col("is_buyer_maker")  # -1 if SELL, 1 if BUY
    )
    df_currency_pair = df_currency_pair.with_columns(
        quantity_sign=pl.col("side_sign") * pl.col("quantity"),
        quote_sign=pl.col("side_sign") * pl.col("quote_abs")
    )
    # Aggregate ticks executed at the same ns timestamp into trades
    df_trades = aggregate_ticks_into_trades(df_currency_pair=df_currency_pair)
    df_trades = compute_slippage(df_trades=df_trades)

    # Now after all preprocessing compute features for CurrencyPair
    # number of aggregated trades within the time interval of df_currency_pair
    num_aggregated_trades: int = df_trades.select(pl.len()).collect().item()

    prices = df_trades.select(pl.col("price_first").first(), pl.col("price_last").last())
    price_first, price_last = [price.item() for price in prices.collect()]

    log_interval_return: float = np.log(price_last / price_first)

    desired_offsets: List[TimeOffset] = [
        TimeOffset.FIVE_SECONDS, TimeOffset.TEN_SECONDS, TimeOffset.HALF_MINUTE, TimeOffset.MINUTE
    ]
    volume_imbalance_features: Dict[str, float] = compute_volume_imbalance(
        df_trades=df_trades, bounds=bounds, time_offsets=desired_offsets
    )
    slippage_features: Dict[str, float] = compute_slippage_features(
        df_trades=df_trades, bounds=bounds, time_offsets=desired_offsets
    )

    return {
        "currency_pair": currency_pair.binance_name,
        "log_interval_return": log_interval_return,
        "num_aggregated_trades": num_aggregated_trades,
        **volume_imbalance_features,
        **slippage_features,
    }
