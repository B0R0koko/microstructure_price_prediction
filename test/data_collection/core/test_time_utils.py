from datetime import date
from typing import List

from core.time_utils import generate_month_time_chunks, generate_daily_time_chunks

expected_result_monthly: List[date] = [
    date(2023, 2, 1),
    date(2023, 3, 1),
    date(2023, 4, 1),
    date(2023, 5, 1),
    date(2023, 6, 1),
    date(2023, 7, 1),
    date(2023, 8, 1),
    date(2023, 9, 1),
    date(2023, 10, 1),
    date(2023, 11, 1),
    date(2023, 12, 1),
    date(2024, 1, 1),
    date(2024, 2, 1)
]


def test_generate_monthly_time_chunks() -> None:
    """Test if we correctly generate year_months dates"""
    test_start_time: date = date(2023, 1, 28)
    test_end_time: date = date(2024, 3, 3)

    generated_monthly_chunks: List[date] = generate_month_time_chunks(
        start_date=test_start_time,
        end_date=test_end_time
    )

    assert generated_monthly_chunks == expected_result_monthly, "Something wrong about how we generate year months dates"


expected_result_daily: List[date] = [
    date(2023, 1, 28),
    date(2023, 1, 29),
    date(2023, 1, 30),
    date(2023, 1, 31),
    # Second half of dates
    date(2024, 3, 1),
    date(2024, 3, 2),
    date(2024, 3, 3)
]


def test_generate_daily_time_chunks() -> None:
    """Test if we correctly generate dates that will be used to download trades data in daily mode"""
    test_start_time: date = date(2023, 1, 28)
    test_end_time: date = date(2024, 3, 3)

    generated_daily_chunks: List[date] = generate_daily_time_chunks(
        start_date=test_start_time,
        end_date=test_end_time
    )

    assert generated_daily_chunks == expected_result_daily, "Something wrong about how we generate year months dates"
