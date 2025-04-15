"""
Prototype interface for reading MERRA2 aerosol optical thickness ancillary data.

As of April 2025, GA does not have a local MERRA2 mirror (e.g. hosted at NCI),
unlike the ECWMF ERA5 data. This partially prevents development of a true MERRA2
solution for `ard-pipeline`.
"""

import datetime
import os

import xarray

TOTAL_AEROSOL_EXTINCTION = "TOTEXTTAU"


# TODO: is user aerosol override handling required?
def aerosol_workflow(merra2_data_dir, acquisition_datetime, lat_longs):
    """
    Top level workflow function to capture MERRA2 aerosol ancillary data.
    """
    aerosol_path = build_merra2_path(merra2_data_dir, acquisition_datetime)
    dataset = xarray.open_dataset(aerosol_path)

    for lat_long in lat_longs:
        aerosol = get_closest_value(dataset, acquisition_datetime, lat_long)
        yield aerosol


# MERRA2 data has file names like:
# MERRA2_300.tavg1_2d_aer_Nx.20080901.nc4
def build_merra2_path(base_dir, date_time: datetime.datetime):
    """
    Build & return expected path to an MERRA 2 NetCDF data file.

    Given acquisition metadata, create expected path containing the ancillary
    data at the acquisition time.

    :param base_dir: Root dir path for MERRA2 data
    :param date_time: acquisition datatime
    """

    year, month, day = date_time.year, date_time.month, date_time.day
    base = f"MERRA2_300.tavg1_2d_aer_Nx.{year}{month:02d}{day:02d}.nc4"

    # TODO: add base dir & vars as required
    path = os.path.join(base_dir, base)
    return path


def get_closest_value(
    xa: xarray.Dataset, date_time: datetime.datetime, lat_long: tuple
):
    """
    Returns closest *previous* value for the variable at the time & location.

    :param xa: an *open* xarray Dataset of MERRA2 NetCDF data
    :param date_time: acquisition datetime
    :param lat_long: (lat, long) tuple of the pixel to extract data for
    """

    aerosol_thickness = xa[TOTAL_AEROSOL_EXTINCTION]
    latitude, longitude = lat_long
    subset = aerosol_thickness.sel(
        time=date_time, method="ffill", lat=latitude, lon=longitude
    )
    return subset.data
