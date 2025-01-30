"""
See NCI `rt52` project for a clone of the data.

This module is a prototype for working with ERA5 reanalysis NetCDF files.
"""

import datetime
import os.path
import typing

PATHNAME_DATE_FORMAT = "%Y%m%d"


# TODO: parse encoded filename metadata to a data structure
#  using NamedTuple as data should be static
# TODO: annotate as dataclass to provide Py style sorting?
class ERA5FileMeta(typing.NamedTuple):
    """
    Metadata from ERA5 pressure level reanalysis files.

    Metadata is stored in file names. Separate parsing here, to reduce complexity
    of ERA5 workflow code.
    """

    variable: str
    dataset: str
    stream: str  # TODO: e.g. 'oper' what is this?
    start_time: datetime.datetime  # datetime to specify start hour of 1st timestep
    stop_time: datetime.datetime  # TODO: hour of last timestep?
    path: str

    @classmethod
    def from_basename(cls, path_basename):
        # parse file meta from "z_era5_oper_pl_20240101-20240131" base name
        base, _ = os.path.splitext(path_basename)  # drop `.nc`
        var, _, _, _, time_range = base.split("_")
        start, stop = time_range.split("-")
        start_tm = datetime.datetime.strptime(start, PATHNAME_DATE_FORMAT)
        stop_tm = datetime.datetime.strptime(stop, PATHNAME_DATE_FORMAT)

        meta = ERA5FileMeta(var, None, None, start_tm, stop_tm, path_basename)
        return meta
