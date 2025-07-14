"""
This prototype module uses ERA5 reanalysis data for ard-pipeline ancillaries.

ERA5 is an alternate ancillary data source to that utilised in the standard NBAR
workflow over Australia, and is used in the DE Antarctica project.

See NCI's `rt52` project for the ERA5 data mirror (in NetCDF).

Usage notes:

The main top level functions are:
* `profile_data_frame_workflow()`
* `ozone_workflow()`

The workflows are designed by *composition* to encapsulate the complexity with
the ERA5 file structure & data access. Typically, only the workflow functions
should be used unless specific use cases require fine-grained control.
"""

# NB: avoid importing wagl.acquisition as it needs Fortran dependencies

import calendar
import datetime
import numbers
import os.path
import typing
import warnings

import pandas as pd
import xarray

import wagl.atmos as atmos
from wagl.constants import GEOPOTENTIAL_HEIGHT, PRESSURE, RELATIVE_HUMIDITY, TEMPERATURE

# strptime format for "YYYYMMDD" strings
PATHNAME_DATE_FORMAT = "%Y%m%d"
MIDNIGHT_1900 = datetime.datetime(1900, 1, 1)  # "0" time for Jan 1900

# See https://en.wikipedia.org/wiki/Standard_gravity
STANDARD_GRAVITY = 9.80665

# Valid ozone data ranges can be found here: https://ozonewatch.gsfc.nasa.gov/
TCO3_MINIMUM_ATM_CM = 0.0
TCO3_MAXIMUM_ATM_CM = 700.0 / 1000  # convert Dobson units to ATM-CM
TCO3_LOW_ATM_CM = 100.0 / 1000


# TODO: annotate as dataclass to provide Py style sorting?
class ERA5FileMeta(typing.NamedTuple):
    """
    Metadata from ERA5 surface & pressure level reanalysis files.

    ERA5 files have metadata within their file names. This class handles parsing
    file naming data to reduce complexity in ERA5 workflows.
    """

    # NB: cannot seem to find an ERA5 data naming standard, but hints are here:
    # https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation and
    # https://docs.dkrz.de/doc/dataservices/finding_and_accessing_data/era_data/index.html
    variable: str
    dataset: str
    stream: str  # TODO: e.g. 'oper' what is this? (operational stream?)
    level_type: str  # either "pl" pressure levels or "sfc" surface for NCI data

    # ERA5 filenames provide start & end dates without a time component.
    start_time: datetime.datetime  # datetime to specify start hour of 1st timestep
    stop_time: datetime.datetime  # TODO: hour of last timestep?
    path: str

    @classmethod
    def from_basename(cls, path_basename):
        """
        Return new ERA5FileMeta instance.

        :param path_basename: the *base* filename with no directory component.
        """
        # parse file meta from "z_era5_oper_pl_20240101-20240131" base name
        base, _ = os.path.splitext(path_basename)  # drop `.nc`
        var, ds, _stream, x, time_range = base.split("_")
        start, stop = time_range.split("-")
        start_tm = datetime.datetime.strptime(start, PATHNAME_DATE_FORMAT)
        stop_tm = datetime.datetime.strptime(stop, PATHNAME_DATE_FORMAT)

        meta = ERA5FileMeta(var, ds, _stream, x, start_tm, stop_tm, path_basename)
        return meta


class ERA5Error(Exception):
    pass


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
    Returns closest *previous* value for the variable at the time & location.

    `xarray` automatically handles data unpacking (scale factor & offsets) when
    extracting variables. Calling code can use the data as is. See:
    https://help.marine.copernicus.eu/en/articles/5470092-how-to-use-add_offset-and-scale_factor-to-calculate-real-values-of-a-variable

    :param xa: an *open* xarray Dataset of ERA5 NetCDF data
    :param variable: name of the ERA5 variable to extract
    :param date_time: acquisition datetime
    :param latlong: (lat, long) tuple of the pixel to extract data for
    """

    # xarray.sel() raises KeyError if date_time & latitude are not within the
    # variable's min-max range. Specifying *longitudes* outside +/-180 does not
    # cause exceptions, however it returns an empty data array.
    # TODO: catch KeyError or break & fail fast to indicate likely bad code?
    #  Mismatching time/location vars are unlikely to occur if using the funcs
    #  to build the ERA5 data paths, this should select the correct NetCDF
    var = xa[variable]
    latitude, longitude = latlong
    subset = var.sel(  # NB: sel() retrieves data for single & multiple levels
        time=date_time, method="ffill", latitude=latitude, longitude=longitude
    )
    return subset.data


ERA5_MULTI_LEVEL_VARIABLES = ("r", "t", "z")

# ERA5 single levels have a variable in the file name & sometimes a different
# variable within the NetCDF file.
ERA5_SINGLE_LEVEL_VARIABLES = ("2t", "z", "sp", "2d")
ERA5_SINGLE_LEVEL_NC_VARIABLES = ("t2m", "z", "sp", "d2m")
ERA5_TOTAL_COLUMN_OZONE = "tco3"


class MultiLevelVars(typing.NamedTuple):
    """
    Container for ERA5 atmospheric values for all above surface layers.

    This convenience class collects ECWMF/ERA5 layered atmospheric data into a
    single location for processing into custom MODTRAN profiles.
    """

    relative_humidity: typing.Sequence
    temperature: typing.Sequence
    geopotential: typing.Sequence


class SingleLevelVars(typing.NamedTuple):
    """
    Container for ERA5 atmospheric values for surface level readings.

    This convenience class collects ECWMF/ERA5 surface level atmospheric data
    into a single location for processing into custom MODTRAN profiles.
    """

    temperature: numbers.Number
    geopotential: numbers.Number
    surface_pressure: numbers.Number
    dewpoint_temperature: numbers.Number


# TODO: make args more explicit for any non prototype code
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

    raw_multi_level = [  # read 37 levels from multi level vars
        get_closest_value(xf, var, date_time, latlong) for var, xf in var_datasets
    ]

    # TODO: check / handle NODATA / fail fast

    # bundle the extracted atmos profile layers
    multi_level_vars = MultiLevelVars(*raw_multi_level)

    # Extract single level/surface vars:
    # 2t -> temperature at 2m
    # z -> geopotential
    # sp -> surface pressure
    # 2d -> dewpoint temperature (2m)
    var_datasets = zip(ERA5_SINGLE_LEVEL_NC_VARIABLES, single_level_datasets)

    raw_single_level = [
        get_closest_value(xf, var, date_time, latlong) for var, xf in var_datasets
    ]

    # TODO: check / handle NODATA / fail fast

    # bundle the extracted atmos profile surface layer vars
    single_level_vars = SingleLevelVars(*raw_single_level)
    return multi_level_vars, single_level_vars


def open_profile_data_files(multi_paths, single_paths):
    """
    Opens single & multi level ERA5 NetCDF files & returns xarray datasets.

    NB: this keeps I/O ops outside data processing functions.
    """
    xf_multi_level_datasets = [xarray.open_dataset(p) for p in multi_paths]
    xf_single_level_datasets = [xarray.open_dataset(p) for p in single_paths]
    return xf_multi_level_datasets, xf_single_level_datasets


def build_era5_path(base_dir, var, date_time: datetime.datetime, single=True):
    """
    Build & return expected path to an ERA5 NetCDF data file.

    Given acquisition metadata, create the expected ERA5 path containing the
    ancillary data at the acquisition time.

    :param base_dir: Root dir path for ERA5 data (e.g. "/g/data/rt53/era5")
    :param var: name of variable of interest
    :param date_time: acquisition datatime
    :param single: True for "single-levels" data, False for "pressure-levels".
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


# TODO: keep I/O out of this function
def build_profile_data_frame(
    multi_level_vars: MultiLevelVars, single_level_vars: SingleLevelVars, ecwmf_levels
):
    """
    Builds MODTRAN atmospheric profile data frames.

    Merges single/surface & multiple level atmospheric data into a 2D table as
    a MODTRAN input. Scaling & re-ordering is performed to ensure the data is
    suitable for MODTRAN requirements (e.g. decreasing pressure with height).

    :param multi_level_vars: a MultiLevelVars instance of the non-surface ERA5
                             atmospheric data layers.
    :param single_level_vars: SingleLevelVars instance of the surface level data.
    :param ecwmf_levels: Sequence of ECWMF pressure levels. This is assumed to
                         be in **increasing** order.
    :return: a profile data frame (table) suitable for MODTRAN.
    """
    rh = atmos.relative_humidity(
        single_level_vars.temperature,
        single_level_vars.dewpoint_temperature,
        kelvin=True,
    )

    # transform column order & scale for MODTRAN
    geopotential_height = reversed(
        scale_z_to_geopotential_height(multi_level_vars.geopotential)
    )

    # Sanitise relative humidity range for MODTRAN
    relative_humidity = multi_level_vars.relative_humidity[::-1]
    relative_humidity = relative_humidity.clip(
        atmos.MIN_RELATIVE_HUMIDITY, atmos.MAX_RELATIVE_HUMIDITY
    )

    temperature = atmos.kelvin_2_celcius(multi_level_vars.temperature)

    # MODTRAN requires monotonically decreasing pressure (ecwmf_levels)
    var_name_mapping = {
        GEOPOTENTIAL_HEIGHT: geopotential_height,
        PRESSURE: reversed(ecwmf_levels),  # switch to decreasing pressure order
        TEMPERATURE: temperature,
        RELATIVE_HUMIDITY: relative_humidity,
    }

    profile_frame = pd.DataFrame(var_name_mapping)

    # apply data scaling & corrections
    surface_pressure = single_level_vars.surface_pressure / 100.0
    geopotential_height = scale_z_to_geopotential_height(single_level_vars.geopotential)
    temperature = atmos.kelvin_2_celcius(single_level_vars.temperature)

    # construct data frame with single level vars as the surface data & multi
    # level data as the rest of the atmospheric values as elevation increases
    # insert surface level parameters, expand rows to 38 levels
    profile_frame.loc[-1] = [
        geopotential_height,
        surface_pressure,
        temperature,
        rh,
    ]

    profile_frame.index = profile_frame.index + 1  # shift index
    profile_frame = profile_frame.sort_index()

    clean_profile_frame = remove_inversions(profile_frame)
    return clean_profile_frame


def scale_z_to_geopotential_height(z, nodata=None):
    """
    Scale geopotential in m**2 s**-2 to kilometres.

    :param z: geopotential value.
    :param nodata: NODATA value
    """
    # >>> ds = xarray.open_dataset(path_to_era5_single_level_geopotential)
    # >>> ds.z.units
    # 'm**2 s**-2'
    #
    # dividing by gravity m/s**2 leaves metres?
    # divide metres by 1000 for kilometres

    if nodata and z == nodata:
        msg = "TODO: handle case where extracted geopotential is NODATA"
        raise NotImplementedError(msg)

    scaled_data = z / STANDARD_GRAVITY / 1000.0
    return scaled_data


def remove_inversions(profile_frame):
    """
    Drop pressure inversion data to clean the atmospheric profile for MODTRAN.

    Assumptions: the function assumes the profile row 0 is surface level data.

    :param profile_frame: Atmospheric profile data frame, see `build_profile_data_frame()`.
    """
    # TODO: does this need to filter on geopotential height too?
    surface_pressure = profile_frame["Pressure"][0]

    # using <= keeps surface pressure row & flags higher pressure layers
    inversions = profile_frame["Pressure"] <= surface_pressure
    return profile_frame[inversions]


def profile_data_frame_workflow(
    era5_data_dir, acquisition_datetime, lat_longs, ecwmf_levels
):
    """
    Top level workflow generator function for per coordinate MODTRAN data frames.

    This generator produces a MODTRAN suitable atmospheric profile data frame for
    each coordinate in `lat_longs`. ERA5 data is assumed to be accessible via a
    filesystem path.

    :param era5_data_dir: path to the root ERA5 data directory.
    :param acquisition_datetime: time of acquisition.
    :param lat_longs: Sequence of (lat, long) coordinate tuples.
    :param ecwmf_levels: see wagl.ancillary.ECWMF_LEVELS.
    """

    # NB: ecwmf_levels is an arg to avoid wagl.ancillary circular import

    multi_paths, single_paths = build_era5_profile_paths(
        era5_data_dir,
        ERA5_MULTI_LEVEL_VARIABLES,
        ERA5_SINGLE_LEVEL_VARIABLES,
        acquisition_datetime,
    )

    # ERA5 data has a ~4 month production lag & may not exist for the acquisition
    # fail fast in this DE Antarctica prototype
    for p in multi_paths + single_paths:
        if not os.path.exists(p):
            msg = (
                f"ERA5 data not found {p}\nIs the ERA5 data missing due to"
                f" production lag?"
            )
            raise FileNotFoundError(msg)

    xf_multi, xf_single = open_profile_data_files(multi_paths, single_paths)

    # use generator to keep files open for multiple data point reads
    # NB: it's possible some reads are repeated for coarse data
    for lat_lon in lat_longs:
        multi_vars, single_vars = profile_data_extraction(
            xf_multi, xf_single, acquisition_datetime, lat_lon
        )

        frame = build_profile_data_frame(multi_vars, single_vars, ecwmf_levels)
        yield frame


# TODO: is a user ozone override required?
def ozone_workflow(era5_data_dir, acquisition_datetime, lat_longs):
    """
    Top level workflow generator to read ERA5 ozone ancillary data.

    Total column ozone (tco3) is read from ERA5 in kg/m2 & is converted to ATM-CM
    atmosphere centimetres for MODTRAN.
    """
    ozone_path = build_era5_path(
        era5_data_dir, ERA5_TOTAL_COLUMN_OZONE, acquisition_datetime, single=True
    )
    dataset = xarray.open_dataset(ozone_path)

    for lat_lon in lat_longs:
        ozone_kgm2 = read_ozone_data(dataset, acquisition_datetime, lat_lon)
        ozone_atm_cm = convert_ozone_atm_cm(ozone_kgm2)

        # NB: assume the tco3 data is (mostly) valid initially.
        #
        # In the absence of expert recommendations, exploratory data analysis
        # work provided a high level data summary. Sampling 2020 to 2025 from
        # -60 to -90 degrees provided the following:
        # * NODATA values were *not* found
        # * NaN was not present
        # * ERA5 reanalysis data did not contain any invalid data.
        # * Minimum ozone was ~100 Dobsons or 0.1 ATM-CM in late winter/spring,
        #   typically hovering around 250 Dobsons for much of the year.
        # * Maximum ozone was ~525 Dobsons, but was typically ~350-400 Dobsons
        # * This range is consistent with the 0-700 Dobson range on NASA's
        #   Ozone Watch site https://ozonewatch.gsfc.nasa.gov/.
        # * Seasonal changes are apparent with ozone.
        # * See https://github.com/OpenDataCubePipelines/ard-pipeline/issues/111#issuecomment-3031092106
        #   for an ozone data plot.
        #
        # Update if expert recommendations are provided for ozone ranges.

        if has_invalid_minimum_ozone_atm_cm(ozone_atm_cm):
            msg = (
                f"{ozone_path} contains invalid zero &/or negative values at "
                f"{lat_lon}. The DE Antarctica prototype has not determined "
                f"handling  requirements for this case yet."
            )
            raise NotImplementedError(msg)

        if has_invalid_maximum_ozone_atm_cm(ozone_atm_cm):
            msg = (
                f"{ozone_path} contains invalid positive tco3 values at {lat_lon}. "
                f"The DE Antarctica prototype has not determined handling"
                f"requirements for this case yet."
            )
            raise NotImplementedError(msg)

        if has_low_minimum_ozone_atm_cm(ozone_atm_cm):
            msg = (
                f"Ozone below 0.1 ATM CM found in {ozone_path} at {lat_lon} on "
                f"{acquisition_datetime}. Check the source data given the very "
                f"low ozone reading."
            )
            warnings.warn(msg)

        yield ozone_atm_cm


def read_ozone_data(
    ozone_dataset: xarray.Dataset,
    acquisition_datetime,
    lat_long: tuple,
):
    """
    Retrieve total column of ozone (tco3) for an acquisition.

    :param ozone_dataset: an open `xarray` of ERA tco3 data.
    :param acquisition_datetime:
    :param lat_long: (latitude, longitude) tuple
    """

    # splitting out this functionality keeps I/O in `ozone_workflow()`
    tco3 = get_closest_value(
        ozone_dataset,
        ERA5_TOTAL_COLUMN_OZONE,
        acquisition_datetime,
        lat_long,
    )

    return tco3


def convert_ozone_atm_cm(tco3_kgm2):
    """
    Convert ERA5 kg/m2 to ATM-CM (atmosphere centimetres).

    MODTRAN requires ATM-CM for its ozone inputs unit.
    """
    # See https://codes.ecmwf.int/grib/param-db/206 for the `tco3` parameter
    # database entry & details of total column ozone

    # The DE Ant prototype does not attempt to detect or handle NODATA. Initial
    # data analysis indicates that `tco3` doesn't contain NODATA, see:
    #  https://github.com/OpenDataCubePipelines/ard-pipeline/issues/111).
    #
    # NCI's converted ERA5 use -32767 (int16) for NODATA, whereas ERA5 GRIB
    # files have an attr of `GRIB_missingValue: 3.4028234663852886e+38`. It's
    # possible a NODATA value is included for completeness.

    return (tco3_kgm2 / 2.1415) * 100


def has_invalid_minimum_ozone_atm_cm(tco3_atm_cm):
    return (tco3_atm_cm <= TCO3_MINIMUM_ATM_CM).any()


def has_low_minimum_ozone_atm_cm(tco3_atm_cm):
    return (
        (TCO3_MINIMUM_ATM_CM < tco3_atm_cm) & (tco3_atm_cm <= TCO3_LOW_ATM_CM)
    ).any()


def has_invalid_maximum_ozone_atm_cm(tco3_atm_cm):
    return (tco3_atm_cm > TCO3_MAXIMUM_ATM_CM).any()
