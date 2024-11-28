from pathlib import Path

from scrapy.crawler import CrawlerProcess

from core.collect_mode import CollectMode
from core.currency import CurrencyPair
from data_collection.datavision.crawler import TradesCrawler
from data_collection.datavision.settings import SETTINGS


def main():
    data_dir: Path = Path("D:/data/zipped_data")

    process: CrawlerProcess = CrawlerProcess(settings=SETTINGS)

    process.crawl(
        TradesCrawler,
        currency_pairs=[CurrencyPair(base="BTC", term="USDT")],
        collect_mode=CollectMode.MONTHLY,
        output_dir=data_dir
    )
    process.start()


if __name__ == "__main__":
    main()
