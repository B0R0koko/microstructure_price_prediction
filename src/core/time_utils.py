from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta, datetime, time
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


def start_of_the_day(day: date) -> datetime:
    """Converts date to datetime with 0:00 time"""
    return datetime.combine(date=day, time=time(hour=0, minute=0, second=0))


def end_of_the_day(day: date) -> datetime:
    """Converts date to datetime with 23:59:59:9999 time"""
    return start_of_the_day(day=day) + timedelta(days=1) - timedelta(microseconds=1)


@dataclass
class Bounds:
    start_inclusive: datetime
    end_exclusive: datetime

    @classmethod
    def for_days(cls, start_date: date, end_date: date):
        return cls(
            start_inclusive=start_of_the_day(day=start_date),
            end_exclusive=end_of_the_day(day=end_date),
        )

    def __str__(self) -> str:
        return f"Bounds: {self.start_inclusive} - {self.end_exclusive}"

    def generate_overlapping_bounds(self, step: timedelta, interval: timedelta) -> List["Bounds"]:
        """Returns a list of bounds created from parent Bounds interval with a certain interval size and step"""
        intervals: List["Bounds"] = []

        lb = self.start_inclusive

        while True:
            rb: datetime = lb + interval
            intervals.append(Bounds(start_inclusive=lb, end_exclusive=rb))  # create new overlapping sub-Bounds
            lb += step

            if rb >= self.end_exclusive:
                break

        return intervals


class TimeOffset(Enum):
    FIVE_SECONDS: timedelta = timedelta(seconds=5)
    TEN_SECONDS: timedelta = timedelta(seconds=10)
    HALF_MINUTE: timedelta = timedelta(seconds=30)
    MINUTE: timedelta = timedelta(minutes=1)
    FIVE_MINUTES: timedelta = timedelta(minutes=5)
    FIFTEEN_MINUTES: timedelta = timedelta(minutes=15)
    HALF_HOUR: timedelta = timedelta(minutes=30)
    HOUR: timedelta = timedelta(hours=1)


if __name__ == "__main__":
    bounds: Bounds = Bounds(
        start_inclusive=start_of_the_day(date.today()),
        end_exclusive=end_of_the_day(date.today()),
    )

    print(bounds)
