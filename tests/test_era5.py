import calendar
import datetime
import os
import socket

import numpy as np
import pytest
import xarray as xr

from wagl import era5

RAW_NUM_LEVELS = 37
TOTAL_NUM_LEVELS = 38  # 36 levels + 1 surface level


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

    try:
        xf = xr.open_dataset(path, engine="h5netcdf")
    except ValueError:
        # try without engine arg...
        xf = xr.open_dataset(path)

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

    try:
        xf = xr.open_dataset(path, engine="h5netcdf")
    except ValueError:
        # try without engine arg...
        xf = xr.open_dataset(path)

    data = era5.find_closest_era5_pressure(
        xf, "z", acquisition_datetime, latlong=mawson_peak_heard_island_lat_lon
    )

    assert data.shape == (37,)  # should just be levels


def test_build_era5_path_single_level(era5_data_dir, acquisition_datetime):
    var = "z"
    single = True
    p = era5.build_era5_path(era5_data_dir, var, acquisition_datetime, single)
    assert p.endswith(
        "single-levels/reanalysis/z/2023/z_era5_oper_sfc_20230201-20230228.nc"
    )


def test_build_era5_path_multi_level(era5_data_dir, acquisition_datetime):
    var = "z"
    single = False
    p = era5.build_era5_path(era5_data_dir, var, acquisition_datetime, single)
    assert p.endswith(
        "pressure-levels/reanalysis/z/2023/z_era5_oper_pl_20230201-20230228.nc"
    )


def test_era5_profile_data_extraction(
    era5_data_dir, acquisition_datetime, mawson_peak_heard_island_lat_lon
):
    multi_paths, single_paths = era5.build_era5_profile_paths(
        era5_data_dir,
        era5.ERA5_MULTI_LEVEL_VARIABLES,
        era5.ERA5_SINGLE_LEVEL_VARIABLES,
        acquisition_datetime,
    )

    xf_multi, xf_single = era5.open_profile_data_files(multi_paths, single_paths)

    rtz, single = era5.profile_data_extraction(
        xf_multi, xf_single, acquisition_datetime, mawson_peak_heard_island_lat_lon
    )

    assert len(rtz) == 3
    # TODO: for var in rtz: assert len(var) == NUM_LEVELS
    assert len(single) == 4


def test_build_profile_data_frame():
    # TODO: read 37 levels from the MLs
    # TODO: read & insert surface values --> is 38 levels
    # TODO: add 6 HARD CODED levels

    # create fake multi level ERA5 data
    relative_humidity_ml = list(range(55, 55 - RAW_NUM_LEVELS, -1))  # descending RH
    temperature_ml = np.array([280 - (i * 5) for i in range(RAW_NUM_LEVELS)])
    geopotential_ml = np.array(
        [2000 + (i * 100) for i in range(RAW_NUM_LEVELS)]
    )  # TODO: copy NetCDF order

    for var in (relative_humidity_ml, temperature_ml, geopotential_ml):
        assert len(var) == RAW_NUM_LEVELS

    multi_level_data = era5.MultiLevelVars(
        relative_humidity_ml, temperature_ml, geopotential_ml
    )

    # create fake single level ERA5 data
    temperature_sl = 285  # NB: start with kelvin
    geopotential_sl = 2300.0
    surface_pressure_sl = 1100.0 * 100  # NB: mimic the units in NetCDF
    dewpoint_temperature_sl = 2270  # NB: start with kelvin

    single_level_data = era5.SingleLevelVars(
        temperature_sl, geopotential_sl, surface_pressure_sl, dewpoint_temperature_sl
    )

    profile_frame = era5.build_profile_data_frame(multi_level_data, single_level_data)
    assert profile_frame is not None

    for key in ("Geopotential_Height", "Pressure", "Temperature", "Relative_Humidity"):
        assert profile_frame[key].size == TOTAL_NUM_LEVELS

    print()
    print(profile_frame)


def test_build_profile_data_frame_real_data(
    era5_data_dir, acquisition_datetime, mawson_peak_heard_island_lat_lon
):
    multi_paths, single_paths = era5.build_era5_profile_paths(
        era5_data_dir,
        era5.ERA5_MULTI_LEVEL_VARIABLES,
        era5.ERA5_SINGLE_LEVEL_VARIABLES,
        acquisition_datetime,
    )

    xf_multi, xf_single = era5.open_profile_data_files(multi_paths, single_paths)

    multi_vars, single_vars = era5.profile_data_extraction(
        xf_multi, xf_single, acquisition_datetime, mawson_peak_heard_island_lat_lon
    )

    frame = era5.build_profile_data_frame(multi_vars, single_vars)
    assert frame is not None

    print()
    print(frame)
