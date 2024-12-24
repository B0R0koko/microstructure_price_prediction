import os
from pathlib import Path
from typing import Optional, Dict, Any

import scrapy

from core.currency import CurrencyPair
from data_collection.datavision.crawler import TradesCrawler


class ZipPipeline:

    def process_item(self, item: Dict[str, Any], spider: TradesCrawler) -> None:
        response: Optional[scrapy.http.Response] = item.get("response")
        currency_pair: Optional[CurrencyPair] = item.get("currency_pair")
        href: Optional[str] = item.get("href")

        assert hasattr(spider, "output_dir"), "output_dir must be supplied in spider"
        assert response, "scrapy.http.Response must be supplied in item that is returned from crawler"
        assert currency_pair, "Currency pair must be supplied in item that is returned from crawler"
        assert href, "Href must be supplied in item that is returned from crawler"

        output_dir: Path = spider.output_dir

        ticker_dir: Path = output_dir.joinpath(currency_pair.name)
        os.makedirs(name=ticker_dir, exist_ok=True)

        slug: str = href.split("/")[-1]  # -> 1INCHBUSD-trades-2021-02-24.zip

        with open(ticker_dir.joinpath(slug), "wb") as file:
            file.write(response.body)
