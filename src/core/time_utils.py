from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta, datetime
from enum import Enum
from typing import Optional, List

import pandas as pd


def get_last_day_month(date_to_round: date) -> date:
    """Gets the last date of the month"""
    day: int = monthrange(year=date_to_round.year, month=date_to_round.month)[1]
    return date(year=date_to_round.year, month=date_to_round.month, day=day)


def get_first_day_month(date_to_round: date) -> date:
    """Returns the first day of the month"""
    return date(year=date_to_round.year, month=date_to_round.month, day=1)


def generate_month_time_chunks(start_date: date, end_date: date) -> Optional[List[date]]:
    """Generate a list of months that lie entirely within given interval of start and end dates"""
    date_months: List[date] = [
        _date.date() for _date in pd.date_range(start_date, end_date, freq="MS", inclusive="both").tolist()
    ]
    # check if the last value is correct
    if not date_months:
        return

    if date_months[-1] == get_first_day_month(end_date):
        date_months.pop(-1)
    return date_months


def _convert_to_dates(dates: pd.DatetimeIndex) -> List[date]:
    return [el.date() for el in dates]


def generate_daily_time_chunks(start_date: date, end_date: date) -> Optional[List[date]]:
    days: List[date] = []

    if start_date != get_first_day_month(start_date):
        days.extend(
            _convert_to_dates(
                pd.date_range(start_date, get_last_day_month(start_date), freq="D", inclusive="both")
            )
        )

    if end_date != get_first_day_month(end_date):
        days.extend(
            _convert_to_dates(
                pd.date_range(get_first_day_month(end_date), end_date, freq="D", inclusive="both")
            )
        )

    return days


@dataclass
class Bounds:
    start_time: datetime
    end_time: datetime


class TimeOffset(Enum):
    FIVE_SECONDS: timedelta = timedelta(seconds=5)
    TEN_SECONDS: timedelta = timedelta(seconds=10)
    HALF_MINUTE: timedelta = timedelta(seconds=30)
    MINUTE: timedelta = timedelta(minutes=1)
    FIVE_MINUTE: timedelta = timedelta(minutes=5)
    FIFTEEN_MINUTE: timedelta = timedelta(minutes=15)
    HALF_HOUR: timedelta = timedelta(minutes=30)
