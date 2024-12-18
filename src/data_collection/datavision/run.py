from datetime import date
from pathlib import Path

from scrapy.crawler import CrawlerProcess

from core.collect_mode import CollectMode
from core.currency import collect_all_currency_pairs
from core.time_utils import Bounds
from data_collection.datavision.crawler import TradesCrawler
from data_collection.datavision.settings import SETTINGS


def main():
    data_dir: Path = Path("D:/data/zipped_data")
    bounds: Bounds = Bounds.for_days(
        start_date=date(2024, 8, 1),
        end_date=date(2024, 11, 30)
    )

    process: CrawlerProcess = CrawlerProcess(settings=SETTINGS)

    process.crawl(
        TradesCrawler,
        bounds=bounds,
        currency_pairs=collect_all_currency_pairs(),
        collect_mode=CollectMode.MONTHLY,
        output_dir=data_dir
    )
    process.start()


if __name__ == "__main__":
    main()
