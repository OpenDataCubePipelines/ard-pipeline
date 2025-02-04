"""
This module is a prototype for working with ERA5 reanalysis NetCDF files on NCI.

See NCI `rt52` project for a clone of the ERA5 data.
"""

import calendar
import datetime
import os.path
import typing

import xarray

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


def get_nearest_previous_hour(date_time):
    """
    Convert a datetime to the number of hours since midnight 1900.
    """
    td = date_time - MIDNIGHT_1900
    hours_since_1900 = int(td.total_seconds() / 60 / 60)  # int() strips partial hours
    return hours_since_1900


# TODO: return single point, multiple points or an array?
#  - should this use wagl's helper funcs to extract points?
#  - e.g. in data.py: data.get_pixel()?
#
# TODO: Which method for closest record in ERA5?
#      - Find closest timestep overall?
#      - Find closest previous timestep?
def find_closest_era5_pressure(
    xa: xarray.Dataset, variable: str, date_time: datetime.datetime
):
    """
    TODO: given a datetime, find closest previous record in the NetCDF file
    TODO: extract pressure levels (or only surface level?)

    xa: an *open* xarray Dataset of the source NetCDF
    date_time: acquisition datetime
    """

    var = xa[variable]

    # TODO: 1st retrieve data across all 37 levels & lat/longs
    subset = var.sel(time=date_time, method="ffill")
    return subset.data
