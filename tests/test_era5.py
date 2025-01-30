import calendar
import datetime

import pytest

from wagl import era5


@pytest.fixture
def z_file_basename():
    return "z_era5_oper_pl_20240101-20240131.nc"


@pytest.fixture
def z_start_datetime():
    start = datetime.datetime(2024, 1, 1)  # start at midnight
    return start


@pytest.fixture
def z_stop_datetime():
    stop = datetime.datetime(2024, 1, 31)  # TODO: end a 23:00?
    return stop


@pytest.fixture
def z_file_path():
    return "/g/data/rt52/era5/pressure-levels/reanalysis/z/2024/z_era5_oper_pl_20240101-20240131.nc"


def test_new_meta(z_start_datetime, z_stop_datetime):
    meta = era5.ERA5FileMeta(
        "z", "era5", "oper", "pl", z_start_datetime, z_stop_datetime, "fake_path"
    )
    assert meta is not None


def test_meta_from_basename(z_file_basename, z_start_datetime, z_stop_datetime):
    meta = era5.ERA5FileMeta.from_basename(z_file_basename)
    assert meta.variable == "z"
    assert meta.start_time == z_start_datetime
    assert meta.stop_time == z_stop_datetime


def test_date_span_january():
    jan2024 = datetime.date(2024, 1, 1)
    span = era5.date_span(jan2024)
    assert span == "20240101-20240131"


def test_date_span_february_non_leap_year():
    feb2023 = datetime.date(2023, 2, 10)
    assert not calendar.isleap(2023)
    span = era5.date_span(feb2023)
    assert span == "20230201-20230228"


def test_date_span_february_leap_year():
    assert calendar.isleap(2024)
    feb2024 = datetime.date(2024, 2, 15)
    span = era5.date_span(feb2024)
    assert span == "20240201-20240229"
