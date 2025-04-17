"""
Prototype interface for reading MERRA2 aerosol optical thickness ancillary data.

As of April 2025, GA does not have a local MERRA2 mirror (e.g. hosted at NCI),
unlike the ECWMF ERA5 data. This partially prevents development of a true MERRA2
solution for `ard-pipeline`.

MERRA2 data improves upon the standard ARD workflow as a subset of global
NetCDF attributes show:

Filename = "MERRA2_300.tavg1_2d_aer_Nx.20080901.nc4" ;
Format = "NetCDF-4/HDF-5" ;
SpatialCoverage = "global" ;
ShortName = "M2T1NXAER" ;
GranuleID = "MERRA2_300.tavg1_2d_aer_Nx.20080901.nc4" ;
Title = "MERRA2 tavg1_2d_aer_Nx: 2d,1-Hourly,Time-averaged,Single-Level,Assimilation,Aerosol Diagnostics" ;

LatitudeResolution = "0.5" ;
LongitudeResolution = "0.625" ;

With sub-degree resolution, a Landsat scene around 2 by 2.5 degrees should have
a different aerosol value for each sample coordinate. This improves upon having
a single aerosol value (or default value) for an entire scene.
"""

import datetime
import os

import xarray

PRODUCT_NAME_VERSION = "M2T1NXAER.5.12.4"  # NB: hardcodes prod & version
TOTAL_AEROSOL_EXTINCTION = "TOTEXTTAU"


# TODO: is user aerosol override handling required?
def aerosol_workflow(merra2_data_dir, acquisition_datetime, lat_longs):
    """
    Top level workflow function to capture MERRA2 aerosol ancillary data.
    """
    aerosol_path = build_merra2_path(merra2_data_dir, acquisition_datetime)

    if not os.path.exists(aerosol_path):
        msg = (
            f"MERRA2 data not found {aerosol_path}\nIs the data collection"
            f"incomplete or the merra2_dir_path setting incorrect?"
        )
        raise FileNotFoundError(msg)

    dataset = xarray.open_dataset(aerosol_path)

    for lat_long in lat_longs:
        aerosol = get_closest_value(dataset, acquisition_datetime, lat_long)
        yield aerosol


# MERRA2 data has file names like:
# MERRA2_300.tavg1_2d_aer_Nx.20080901.nc4
def build_merra2_path(base_dir, date_time: datetime.datetime):
    """
    Build & return expected path to an MERRA2 NetCDF data file.

    Given acquisition metadata, create expected path containing the ancillary
    data at the acquisition time.

    :param base_dir: Root dir path for MERRA2 data
    :param date_time: acquisition datatime
    """

    year, month, day = date_time.year, date_time.month, date_time.day
    basename = f"MERRA2_300.tavg1_2d_aer_Nx.{year}{month:02d}{day:02d}.nc4"

    # Add the variable/year/month dir structure
    root_dir = f"{base_dir}/{PRODUCT_NAME_VERSION}/{year}/{month:02d}"
    path = os.path.join(root_dir, basename)
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
