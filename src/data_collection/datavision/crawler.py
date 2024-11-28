# pylint: disable=use-dict-literal

import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from urllib.parse import urlencode

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.http import Request
from scrapy.http.response import Response

from core.collect_mode import CollectMode
from core.currency import CurrencyPair
from data_collection.datavision.settings import SETTINGS

BINANCE_S3: str = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BINANCE_DATAVISION: str = "https://data.binance.vision"


def get_currency_url(
        currency_pair: CurrencyPair,
        collect_mode: CollectMode,
        marker: Optional[str] = None
) -> str:
    """Create url which we use to query binance data vision for the first time for each ticker"""
    params: Dict[str, str] = {
        "delimiter": "/",
        # data/spot/daily/trades/1INCHBUSD/
        "prefix": f"data/spot/{collect_mode.lower()}/trades/{currency_pair.binance_name}/",
    }

    if marker:
        params["marker"] = marker

    datavision_url: str = f"{BINANCE_S3}?{urlencode(params)}"
    return datavision_url


def get_zip_file_url(href: str) -> str:
    """Returns a formatted url string which leads to a zip file with trades data"""
    # https://data.binance.vision/data/spot/monthly/trades/1INCHUSDT/1INCHUSDT-trades-2024-05.zip
    return f"{BINANCE_DATAVISION}/{href}"


class TradesCrawler(scrapy.Spider):
    name = "ticker_crawler"

    def __init__(
            self, currency_pairs: List[CurrencyPair], collect_mode: CollectMode, output_dir: Path, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.currency_pairs: List[CurrencyPair] = currency_pairs
        self.collect_mode: CollectMode = collect_mode
        self.output_dir: Path = output_dir

    def start_requests(self) -> scrapy.Request:
        for currency_pair in self.currency_pairs:
            yield Request(
                url=get_currency_url(currency_pair, collect_mode=self.collect_mode),
                callback=self.parse_currency_pair,  # type:ignore
                meta={"currency_pair": currency_pair, "href_container": []},  # mutable object
            )

    def parse_currency_pair(self, response: Response):
        """Parse hrefs with zip files from currency_pair page"""

        currency_pair: Optional[CurrencyPair] = response.meta.get("currency_pair")
        href_container: List[str] = response.meta.get("href_container")

        assert currency_pair, "Currency pair must be supplied in scrapy.http.Response.meta"
        assert href_container is not None, "Href container must be supplied in scrapy.http.Response.meta"

        hrefs: List[str] = re.findall(pattern=r"<Key>(.*?)</Key>", string=response.text)
        hrefs: List[str] = [href for href in hrefs if "CHECKSUM" not in href]

        href_container.extend(hrefs)

        # if len is 500, then we need to send another request with marker param which is the last entry in hrefs
        if len(hrefs) == 500:
            yield scrapy.Request(
                url=get_currency_url(currency_pair=currency_pair, collect_mode=self.collect_mode, marker=hrefs[-1]),
                callback=self.parse_currency_pair,  # call itself one more time
                meta={"currency_pair": currency_pair, "href_container": href_container},
            )
        # Once we have collected all hrefs into response.meta.href_container we loop over it and send requests that
        # collect zip files

        for href in href_container:
            yield scrapy.Request(
                url=get_zip_file_url(href=href),
                callback=self.parse_response_from_zip_endpoint,
                meta={"href": href, "currency_pair": currency_pair},
            )

    @staticmethod
    def parse_response_from_zip_endpoint(response: Response) -> Dict[str, Any]:
        return {
            "response": response,
            "href": response.meta.get("href"),
            "currency_pair": response.meta.get("currency_pair"),
        }


if __name__ == "__main__":
    data_dir: Path = Path("D:/data")

    process: CrawlerProcess = CrawlerProcess(
        settings=SETTINGS
    )

    process.crawl(
        TradesCrawler,
        currency_pairs=[CurrencyPair("ADA", "USDT")],
        collect_mode=CollectMode.MONTHLY,
        output_dir=data_dir
    )
    process.start()
