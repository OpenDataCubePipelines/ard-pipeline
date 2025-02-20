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


# guard block: check if platform has ERA5 data
# HACK: use temporary env var to point to local file to get test working
env_key = "ERA5_TEMP_DATA_DIR"
platform_err = "Platform lacks ERA5 data"

_data_conds = [
    os.path.exists("/g/data/rt52/era5"),
    env_key in os.environ and os.path.exists(os.environ[env_key]),
]

SYS_MISSING_ERA5_DATA = not any(_data_conds)


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


@pytest.mark.skipif(SYS_MISSING_ERA5_DATA, reason=platform_err)
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

    data = era5.get_closest_value(
        xf, "t2m", acquisition_datetime, latlong=mawson_peak_heard_island_lat_lon
    )

    assert data
    assert float(data)  # HACK: silly test


@pytest.mark.skipif(SYS_MISSING_ERA5_DATA, reason=platform_err)
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

    data = era5.get_closest_value(
        xf, "z", acquisition_datetime, latlong=mawson_peak_heard_island_lat_lon
    )

    assert data.shape == (37,)  # should just be levels


@pytest.fixture
def acquisition_datetime_2011():
    # create fake acquisition_datetime
    return datetime.datetime(2011, 7, 8, 10, 45)


@pytest.fixture
def z_02_2023(era5_data_dir):
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

    return xf


@pytest.mark.skipif(SYS_MISSING_ERA5_DATA, reason=platform_err)
def test_find_closest_value_datetime_outside_range(
    z_02_2023, acquisition_datetime_2011, mawson_peak_heard_island_lat_lon
):
    with pytest.raises(KeyError):
        era5.get_closest_value(
            z_02_2023,
            "z",
            acquisition_datetime_2011,
            latlong=mawson_peak_heard_island_lat_lon,
        )


@pytest.mark.skipif(SYS_MISSING_ERA5_DATA, reason=platform_err)
def test_find_closest_value_positive_latitude_outside_range(
    z_02_2023, acquisition_datetime
):
    for bad_lat in (95, 100, 101):
        assert bad_lat not in z_02_2023.latitude.data

        with pytest.raises(KeyError):
            era5.get_closest_value(
                z_02_2023, "z", acquisition_datetime, latlong=(bad_lat, 130)
            )


@pytest.mark.skip(reason="Determine why xarray.sel() doesn't fail")
@pytest.mark.skipif(SYS_MISSING_ERA5_DATA, reason=platform_err)
def test_find_closest_value_negative_latitude_outside_range(
    z_02_2023, acquisition_datetime
):
    for bad_lat in (-91, -95, -101):
        assert bad_lat not in z_02_2023.latitude.data

        with pytest.raises(KeyError):
            era5.get_closest_value(
                z_02_2023, "z", acquisition_datetime, latlong=(bad_lat, 130)
            )


@pytest.mark.skip(reason="Determine why xarray.sel() doesn't fail")
@pytest.mark.skipif(SYS_MISSING_ERA5_DATA, reason=platform_err)
def test_find_closest_value_positive_longitude_outside_range(
    z_02_2023, acquisition_datetime
):
    for bad_long in (181, 185, 190):
        assert bad_long not in z_02_2023.longitude.data

        with pytest.raises(KeyError):
            era5.get_closest_value(
                z_02_2023, "z", acquisition_datetime, latlong=(79.0, bad_long)
            )


@pytest.mark.skipif(SYS_MISSING_ERA5_DATA, reason=platform_err)
def test_find_closest_value_negative_longitude_outside_range(
    z_02_2023, acquisition_datetime
):
    for bad_long in (-182, -186, -189):
        assert bad_long not in z_02_2023.longitude.data

        with pytest.raises(KeyError):
            era5.get_closest_value(
                z_02_2023, "z", acquisition_datetime, latlong=(69.0, bad_long)
            )


def test_build_era5_path_single_level(acquisition_datetime):
    var = "z"
    single = True
    p = era5.build_era5_path("fake_dir_path", var, acquisition_datetime, single)
    assert p.endswith(
        "single-levels/reanalysis/z/2023/z_era5_oper_sfc_20230201-20230228.nc"
    )


def test_build_era5_path_multi_level(acquisition_datetime):
    var = "z"
    single = False
    p = era5.build_era5_path("fake_dir_path", var, acquisition_datetime, single)
    assert p.endswith(
        "pressure-levels/reanalysis/z/2023/z_era5_oper_pl_20230201-20230228.nc"
    )


@pytest.fixture
def default_profile_paths(era5_data_dir, acquisition_datetime):
    """TODO"""
    multi_paths, single_paths = era5.build_era5_profile_paths(
        era5_data_dir,
        era5.ERA5_MULTI_LEVEL_VARIABLES,
        era5.ERA5_SINGLE_LEVEL_VARIABLES,
        acquisition_datetime,
    )

    for path in multi_paths:
        assert os.path.exists(path), f"{path} does not exist"

    for path in single_paths:
        assert os.path.exists(path), f"{path} does not exist"

    return multi_paths, single_paths


@pytest.mark.skipif(SYS_MISSING_ERA5_DATA, reason=platform_err)
def test_era5_profile_data_extraction(
    era5_data_dir,
    acquisition_datetime,
    mawson_peak_heard_island_lat_lon,
    default_profile_paths,
):
    multi_paths, single_paths = default_profile_paths
    xf_multi, xf_single = era5.open_profile_data_files(multi_paths, single_paths)

    rtz, single = era5.profile_data_extraction(
        xf_multi, xf_single, acquisition_datetime, mawson_peak_heard_island_lat_lon
    )

    assert rtz.relative_humidity.shape == (RAW_NUM_LEVELS,)
    assert rtz.temperature.shape == (RAW_NUM_LEVELS,)
    assert rtz.geopotential.shape == (RAW_NUM_LEVELS,)

    assert len(single) == 4
    assert single.temperature is not None
    assert single.geopotential is not None


def test_build_profile_data_frame():
    # test building a profile data frame with fake data

    # create fake multi level ERA5 data
    relative_humidity_ml = list(range(55, 55 - RAW_NUM_LEVELS, -1))  # descending RH
    temperature_ml = np.array([280 - (i * 5) for i in range(RAW_NUM_LEVELS)])
    geopotential_ml = np.array(
        [2000 + (i * 100) for i in range(RAW_NUM_LEVELS)]
    )  # TODO: copy NetCDF data order

    for var in (relative_humidity_ml, temperature_ml, geopotential_ml):
        assert len(var) == RAW_NUM_LEVELS

    multi_level_data = era5.MultiLevelVars(
        relative_humidity_ml, temperature_ml, geopotential_ml
    )

    # create fake single level ERA5 data
    temperature_sl = 285.0  # NB: start with kelvin
    geopotential_sl = 2300.0
    surface_pressure_sl = 1100.0 * 100  # NB: mimic the units in NetCDF
    dewpoint_temperature_sl = 2270.0  # NB: start with kelvin

    single_level_data = era5.SingleLevelVars(
        temperature_sl, geopotential_sl, surface_pressure_sl, dewpoint_temperature_sl
    )

    profile_frame = era5.build_profile_data_frame(multi_level_data, single_level_data)
    assert profile_frame is not None

    for key in ("Geopotential_Height", "Pressure", "Temperature", "Relative_Humidity"):
        assert profile_frame[key].size == TOTAL_NUM_LEVELS

    print()
    print(profile_frame)


@pytest.mark.skipif(SYS_MISSING_ERA5_DATA, reason=platform_err)
def test_build_profile_data_frame_real_data(
    era5_data_dir,
    acquisition_datetime,
    mawson_peak_heard_island_lat_lon,
    default_profile_paths,
):
    multi_paths, single_paths = default_profile_paths
    xf_multi, xf_single = era5.open_profile_data_files(multi_paths, single_paths)

    multi_vars, single_vars = era5.profile_data_extraction(
        xf_multi, xf_single, acquisition_datetime, mawson_peak_heard_island_lat_lon
    )

    frame = era5.build_profile_data_frame(multi_vars, single_vars)
    assert frame is not None

    print()
    print(frame)


def test_scale_z_to_geopotential_height():
    z = -0.973126067823614
    gph = (z / era5.STANDARD_GRAVITY) / 1000.0  # NB: duplicates scaling func
    res = era5.scale_z_to_geopotential_height(z)
    assert res == gph


def test_scale_z_to_geopotential_height_nodata_fail():
    nodata = -32767
    z = nodata

    with pytest.raises(NotImplementedError):
        era5.scale_z_to_geopotential_height(z, nodata)


@pytest.fixture
def ozone_dataset(era5_data_dir):
    """Return Feb 2023 'tco3' / total column ozone as an open xarray dataset."""
    part_path = (
        "single-levels/reanalysis/tco3/2023/tco3_era5_oper_sfc_20230201-20230228.nc"
    )
    path = os.path.join(era5_data_dir, part_path)
    dataset = xr.open_dataset(path)
    return dataset


@pytest.fixture
def geopotential_dataset(era5_data_dir):
    part = "pressure-levels/reanalysis/z/2023/z_era5_oper_pl_20230201-20230228.nc"
    path = os.path.join(era5_data_dir, part)
    dataset = xr.open_dataset(path)
    return dataset


@pytest.fixture
def t0_02_2023():
    return datetime.datetime.fromisoformat("2023-02-01T00")


@pytest.mark.skipif(SYS_MISSING_ERA5_DATA, reason=platform_err)
def test_verify_xarray_unpacking(ozone_dataset, t0_02_2023):
    """
    Confirm Py environment `xarray` automatically unpacks variables.

    'Unpacking' describes the process of extracting `packed` data, such as 16 bit
    values & applying a scaling factor & offset to convert to a real value.

    The following article indicates xarray automatically handles conversions:
    https://help.marine.copernicus.eu/en/articles/5470092-how-to-use-add_offset-and-scale_factor-to-calculate-real-values-of-a-variable
    """

    # dimension params for 1st ozone value in tco3/ozone dataset
    longitude = -180
    latitude = 90

    ds_from_xarray = ozone_dataset.sel(
        longitude=longitude, latitude=latitude, time=t0_02_2023
    )

    tco3_from_xarray = ds_from_xarray.tco3.data

    # manually calculate unpacked value
    scale_factor = 1.28263439555932e-07
    add_offset = 0.00884041296233444
    tco3_packed = -5879  # manually copied from `ncdump` output
    tco3_unpacked = (tco3_packed * scale_factor) + add_offset

    assert np.allclose(tco3_unpacked, tco3_from_xarray)


@pytest.mark.skipif(SYS_MISSING_ERA5_DATA, reason=platform_err)
def test_verify_xarray_skip_unpacking(geopotential_dataset, t0_02_2023):
    """
    Confirm `xarray` only extracts variables without scale_factor/offset attrs.
    """

    # dimension params for 1st value of z
    longitude = -180
    latitude = 90

    ds_from_xarray = geopotential_dataset.sel(
        longitude=longitude, latitude=latitude, time=t0_02_2023
    )

    z_from_xarray = ds_from_xarray.z.data[0]
    z_from_ncdump = 457292.8  # hand copied into test
    assert np.allclose(z_from_ncdump, z_from_xarray)


@pytest.mark.skipif(SYS_MISSING_ERA5_DATA, reason=platform_err)
def test_read_ozone_data_from_era5_netcdf(
    ozone_dataset, acquisition_datetime, mawson_peak_heard_island_lat_lon
):
    # NB: read_ozone_data() fun is barely required as it's a minor customisation
    #  of get_closest_value(). Keep for readability???
    tco3 = era5.read_ozone_data(
        ozone_dataset,
        acquisition_datetime,
        mawson_peak_heard_island_lat_lon,
    )

    assert tco3 is not None
    assert float(tco3)  # FIXME: rubbish test


# NB: comment this until it's known if the override is needed
# def test_get_ozone_data_from_override():
#     user_override = 0.000111
#     ozone = {
#         "user": user_override
#     }  # use instead of OzoneDict which requires more imports
#     tco3, _ = era5.get_ozone_data_user_override(ozone)
#     assert tco3 == user_override
