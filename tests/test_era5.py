import calendar
import datetime
import os
import socket

import pytest
import xarray as xr

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


@pytest.fixture
def acquisition_datetime():
    # create fake acquisition_datetime
    return datetime.datetime(2023, 2, 4, 11, 23, 45)


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


def test_find_closest_era5_path(acquisition_datetime):
    # assume single directory search as year is known & can be easily located
    # within the NCI /g/data/ file hierarchy
    paths = [
        "z_era5_oper_pl_20230101-20230131.nc",
        "z_era5_oper_pl_20230201-20230228.nc",
        "z_era5_oper_pl_20230301-20230331.nc",
        "z_era5_oper_pl_20230401-20230430.nc",
        "z_era5_oper_pl_20230501-20230531.nc",
        "z_era5_oper_pl_20230601-20230630.nc",
        "z_era5_oper_pl_20230701-20230731.nc",
        "z_era5_oper_pl_20230801-20230831.nc",
        "z_era5_oper_pl_20230901-20230930.nc",
        "z_era5_oper_pl_20231001-20231031.nc",
        "z_era5_oper_pl_20231101-20231130.nc",
        "z_era5_oper_pl_20231201-20231231.nc",
    ]

    span = era5.date_span(acquisition_datetime)
    closest = era5.find_closest_era5_path(paths, span)
    assert closest == paths[1]


def test_get_nearest_previous_hour(acquisition_datetime):
    # data manually copied from z_era5_oper_pl_20230201-20230228.nc
    first_hour = 1078944  # hours since midnight 1900 to midnight 1/2/2023
    three_days = 24 * 3  # delta to midnight 4/2/2023
    offset_11am = 11  # delta for the partial day
    expected = first_hour + three_days + offset_11am
    hour = era5.get_nearest_previous_hour(acquisition_datetime)
    assert hour == expected


@pytest.fixture
def mawson_peak_heard_island_lat_lon():
    """Return (lat, lon) for Mawson Peak on Heard Island."""
    return -53.1046, 73.51710


@pytest.fixture
def era5_data_dir():
    """
    Return dir path to use for temporary testing.

    ERA5 dir trees are like:
    /home/user/data/rt52/era5/pressure-levels/reanalysis/z/2023/

    Return a root path: "/home/user/data/rt52/era5"
    """

    # TODO: simulate or load a real NetCDF file...
    #  problem: each ERA5 multi-level is ~35GB in size!
    #  design data interface or mock xarray to avoid using a real file?

    # HACK: this is NCI platform specific
    if "gadi" in socket.gethostname():
        return "/g/data/rt52/era5"

    # HACK: use temporary env var to point to local file to get test working
    env_key = "ERA5_TEMP_DATA_DIR"

    # HACK: set env var to *local* copy of z_era5_oper_pl_20230201-20230228.nc
    if env_key not in os.environ:
        msg = (
            f"WARN: temporary prototyping code. Set {env_key} to *local* data dir"
            f" of z_era5_oper_pl_20230201-20230228.nc"
        )
        raise NotImplementedError(msg)

    path = os.environ[env_key]
    assert os.path.exists(path)
    return path


def test_find_closest_era5_pressure_single_level(
    era5_data_dir, acquisition_datetime, mawson_peak_heard_island_lat_lon
):
    path = os.path.join(
        era5_data_dir,
        "single-levels/reanalysis/2t/2023",
        "2t_era5_oper_sfc_20230201-20230228.nc",
    )

    xf = xr.open_dataset(path, engine="h5netcdf")
    data = era5.find_closest_era5_pressure(
        xf, "t2m", acquisition_datetime, latlong=mawson_peak_heard_island_lat_lon
    )

    assert data
    assert float(data)  # HACK: silly test


def test_find_closest_era5_pressure_multi_level(
    era5_data_dir, acquisition_datetime, mawson_peak_heard_island_lat_lon
):
    path = os.path.join(
        era5_data_dir,
        "pressure-levels/reanalysis/z/2023",
        "z_era5_oper_pl_20230201-20230228.nc",
    )

    xf = xr.open_dataset(path, engine="h5netcdf")
    data = era5.find_closest_era5_pressure(
        xf, "z", acquisition_datetime, latlong=mawson_peak_heard_island_lat_lon
    )

    assert data.shape == (37,)  # should just be levels
