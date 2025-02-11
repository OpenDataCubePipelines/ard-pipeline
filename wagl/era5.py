"""
This module is a prototype for working with ERA5 reanalysis NetCDF files on NCI.

See NCI `rt52` project for a clone of the ERA5 data.
"""

import calendar
import datetime
import os.path
import typing

import pandas as pd
import xarray

import wagl.atmos as atmos

# strptime format for "YYYYMMDD" strings
PATHNAME_DATE_FORMAT = "%Y%m%d"
MIDNIGHT_1900 = datetime.datetime(1900, 1, 1)  # "0" time for Jan 1900


# TODO: parse encoded filename metadata to a data structure
#  using NamedTuple as data should be static
# TODO: annotate as dataclass to provide Py style sorting?
class ERA5FileMeta(typing.NamedTuple):
    """
    Metadata from ERA5 pressure level reanalysis files.

    ERA5 files have metadata within their file names. This class handles parsing
    file naming data to reduce complexity in ERA5 workflows.
    """

    variable: str
    dataset: str
    stream: str  # TODO: e.g. 'oper' what is this?
    unknown: str  # TODO: rename

    # TODO: filenames provide start & end dates, without a time component. Could
    #  test the hour fields are correct in several NetCDF files & default the
    #  start *datetime* to midnight & stop time as 23:00.
    # TODO: probably need datetime.datetime to select the closest NetCDF data
    #  record to each Acquisition's datetime.
    start_time: datetime.datetime  # datetime to specify start hour of 1st timestep
    stop_time: datetime.datetime  # TODO: hour of last timestep?
    path: str

    @classmethod
    def from_basename(cls, path_basename):
        # parse file meta from "z_era5_oper_pl_20240101-20240131" base name
        base, _ = os.path.splitext(path_basename)  # drop `.nc`
        var, ds, _stream, x, time_range = base.split("_")
        start, stop = time_range.split("-")
        start_tm = datetime.datetime.strptime(start, PATHNAME_DATE_FORMAT)
        stop_tm = datetime.datetime.strptime(stop, PATHNAME_DATE_FORMAT)

        meta = ERA5FileMeta(var, ds, _stream, x, start_tm, stop_tm, path_basename)
        return meta


def date_span(date_obj):
    # Return "YYYYMMDD-YYYYMMDD" string given the year & month
    year = date_obj.year
    month = date_obj.month
    _, last_day = calendar.monthrange(year, month)
    span = f"{year}{month:02}01-{year}{month:02}{last_day:02}"
    return span


def find_closest_era5_path(paths, span):
    # use span to pattern match against sequence of file paths
    result = [p for p in paths if span in p]

    if not result:
        msg = f"No matching path found for {span}"
        raise ValueError(msg)  # TODO: is this the most appropriate error?

    # guard against multiple path matches
    if len(result) == 1:
        return result[0]

    msg = f"Multiple path matches found for {span}"
    raise ValueError(msg)


# TODO: Which method for closest record in ERA5?
#   Nearest timestep or closest previous timestep?
def get_closest_value(
    xa: xarray.Dataset, variable: str, date_time: datetime.datetime, latlong: tuple
):
    """
    Returns closest *previous* value for the given

    :param xa: an *open* xarray Dataset of ERA5 NetCDF data
    :param variable: name of the ERA5 variable to extract
    :param date_time: acquisition datetime
    :param latlong: (lat, long) tuple of the pixel to extract data for
    """

    # NB: sel() retrieves data for single & multiple levels
    var = xa[variable]
    latitude, longitude = latlong
    subset = var.sel(
        time=date_time, method="ffill", latitude=latitude, longitude=longitude
    )
    return subset.data


ERA5_MULTI_LEVEL_VARIABLES = ("r", "t", "z")

# ERA5 single levels have a variable in the file name & sometimes a different
# variable within the NetCDF file.
ERA5_SINGLE_LEVEL_VARIABLES = ("2t", "z", "sp", "2d")
ERA5_SINGLE_LEVEL_NC_VARIABLES = ("t2m", "z", "sp", "d2m")


class MultiLevelVars(typing.NamedTuple):
    relative_humidity: typing.Sequence
    temperature: typing.Sequence
    geopotential: typing.Sequence


class SingleLevelVars(typing.NamedTuple):
    temperature: typing.Sequence
    geopotential: typing.Sequence
    surface_pressure: typing.Sequence
    dewpoint_temperature: typing.Sequence


def profile_data_extraction(
    multi_level_datasets,
    single_level_datasets,
    date_time: datetime.datetime,
    latlong: tuple,
):
    """
    Returns MODTRAN profile data extracted from ERA5 NetCDF files.

    Note the dataset args must provide open files in a specific order. Files are
    not opened in this function to avoid I/O & facilitate unit testing.

    :param multi_level_datasets: *open* xarray.Datasets in order: 'r', 't' & 'z'
    :param single_level_datasets:  *open* xarray.Datasets in order '2t', 'z', 'sp' & '2d'
    :param date_time: Acquisition date/time
    :param latlong: tuple of floats for (lat, long) location.
    """

    # Extract & package these multi level vars:
    # r -> relative humidity
    # t -> temperature
    # z -> geopotential
    var_datasets = zip(ERA5_MULTI_LEVEL_VARIABLES, multi_level_datasets)

    raw_multi_level = [
        get_closest_value(xf, var, date_time, latlong) for var, xf in var_datasets
    ]

    multi_level_vars = MultiLevelVars(*raw_multi_level)

    # Extract single level vars:
    # 2t -> temperature at 2m
    # z -> geopotential
    # sp -> surface pressure
    # 2d -> dewpoint temperature (2m)
    var_datasets = zip(ERA5_SINGLE_LEVEL_NC_VARIABLES, single_level_datasets)

    raw_single_level = [
        get_closest_value(xf, var, date_time, latlong) for var, xf in var_datasets
    ]

    single_level_vars = SingleLevelVars(*raw_single_level)

    # TODO: add to names tuple, named tuple, dict or return multiple values?
    return multi_level_vars, single_level_vars


def open_profile_data_files(multi_paths, single_paths):
    xf_multi_level_datasets = [xarray.open_dataset(p) for p in multi_paths]
    xf_single_level_datasets = [xarray.open_dataset(p) for p in single_paths]
    return xf_multi_level_datasets, xf_single_level_datasets


def build_era5_path(base_dir, var, date_time: datetime.datetime, single=True):
    """
    TODO: return a path to ERA5 file
    """
    type_dir = "single-levels" if single else "pressure-levels"
    _type = "sfc" if single else "pl"
    span = date_span(date_time)
    base = f"{var}_era5_oper_{_type}_{span}.nc"
    path = f"{base_dir}/{type_dir}/reanalysis/{var}/{date_time.year}/{base}"
    return path


def build_era5_profile_paths(
    base_dir, multi_level_vars, single_level_vars, date_time: datetime.datetime
):
    """
    TODO: build all paths for all ERA5 files required to build MODTRAN profiles.
    """
    multi_paths = [
        build_era5_path(base_dir, v, date_time, False) for v in multi_level_vars
    ]
    single_paths = [build_era5_path(base_dir, v, date_time) for v in single_level_vars]
    return multi_paths, single_paths


# HACK: copied from ancillary.py (can't import wagl module without the F90 being built)
ECWMF_LEVELS = [
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
    50,
    70,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    775,
    800,
    825,
    850,
    875,
    900,
    925,
    950,
    975,
    1000,
]


# TODO: keep I/O out of this function
def build_profile_data_frame(
    multi_level_vars: MultiLevelVars, single_level_vars: SingleLevelVars
):
    rh = atmos.relative_humdity(
        single_level_vars.temperature,
        single_level_vars.dewpoint_temperature,
        kelvin=True,
    )

    # transform data for column order & scaling
    geopotential = reversed(scale_geopotential(multi_level_vars.geopotential))
    relative_humidity = reversed(multi_level_vars.relative_humidity)
    temperature = atmos.kelvin_2_celcius(multi_level_vars.temperature)

    var_name_mapping = {
        "Geopotential_Height": geopotential,
        "Pressure": reversed(ECWMF_LEVELS),
        "Temperature": temperature,
        "Relative_Humidity": relative_humidity,
    }

    profile_frame = pd.DataFrame(var_name_mapping)

    # apply data scaling & corrections
    surface_pressure = single_level_vars.surface_pressure / 100.0
    geopotential = scale_geopotential(single_level_vars.geopotential)
    temperature = atmos.kelvin_2_celcius(single_level_vars.temperature)

    # insert surface level parameters
    profile_frame.loc[-1] = [
        geopotential,
        surface_pressure,
        temperature,
        rh,
    ]

    profile_frame.index = profile_frame.index + 1  # shift index
    profile_frame = profile_frame.sort_index()
    return profile_frame


def scale_geopotential(data):
    scaled_data = data / 9.80665 / 1000.0
    return scaled_data


def profile_data_frame_workflow(era5_data_dir, acquisition_datetime, lat_lon):
    """
    TODO: describe overall workflow
    """

    multi_paths, single_paths = build_era5_profile_paths(
        era5_data_dir,
        ERA5_MULTI_LEVEL_VARIABLES,
        ERA5_SINGLE_LEVEL_VARIABLES,
        acquisition_datetime,
    )

    xf_multi, xf_single = open_profile_data_files(multi_paths, single_paths)

    multi_vars, single_vars = profile_data_extraction(
        xf_multi, xf_single, acquisition_datetime, lat_lon
    )

    frame = build_profile_data_frame(multi_vars, single_vars)
    return frame
