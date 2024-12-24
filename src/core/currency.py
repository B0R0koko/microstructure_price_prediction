from dataclasses import dataclass
from typing import List, Dict, Any

import requests


@dataclass
class Currency:
    base: str
    term: str


@dataclass
class CurrencyPair:
    base: str
    term: str

    @classmethod
    def from_string(cls, symbol: str):
        """Parse CurrencyPair from string formatted like this: ADA-USDT"""
        base, term = symbol.split("-")
        return cls(base=base, term=term)

    def __str__(self) -> str:
        return f"{self.base}-{self.term}"

    @property
    def name(self) -> str:
        return f"{self.base}-{self.term}"

    @property
    def binance_name(self) -> str:
        return f"{self.base}{self.term}"


def collect_all_currency_pairs() -> List[CurrencyPair]:
    """Collect a set of all CurrencyPairs traded on Binance"""
    resp = requests.get("https://api.binance.com/api/v3/exchangeInfo")
    data: Dict[str, Any] = resp.json()
    return [
        CurrencyPair(base=entry["baseAsset"], term=entry["quoteAsset"]) for entry in data["symbols"]
    ]
