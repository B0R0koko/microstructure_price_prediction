from typing import List

TRADE_ID: str = "trade_id"
PRICE: str = "price"
QUANTITY: str = "quantity"
QUOTE_QUANTITY: str = "quote_quantity"
TRADE_TIME: str = "trade_time"
IS_BUYER_MAKER: str = "is_buyer_maker"
IS_BEST_MATCH: str = "is_best_match"

BINANCE_TRADE_COLS: List[str] = [
    TRADE_ID, PRICE, QUANTITY, QUOTE_QUANTITY, TRADE_TIME, IS_BUYER_MAKER, IS_BEST_MATCH
]

SYMBOL: str = "symbol"
