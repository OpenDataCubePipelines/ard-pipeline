"""
Tests for reading MERRA-2 aerosol ancillary data.

to run the tests, run pytest like:
$ MERRA2_DATA_DIR="/some/dir/merra2_root"  pytest tests/test_merra2.py

Where MERRA2_DATA_DIR is a directory with the M2T1NXAER.5.12.4 variable & sub
structure of year/month dirs.

If testing on a non-NCI system, copy gadi:/g/data/da82/AODH/USGS/L1/Landsat/C1/092_084/...
    ... LT50920842008269ASA00/LT05_L1TP_092084_20080925_20161029_01_T1.tar
to a directory like /home/user/data/M2T1NXAER.5.12.4/2008/09/ and use that for
the environment variable.
"""

import datetime
import os

import pytest
import xarray

from wagl import merra2

merra2_data_key = "MERRA2_DATA_DIR"

if merra2_data_key in os.environ:
    # assume local testing with a local copy of the data
    MERRA2_DATA_DIR = os.environ[merra2_data_key]
else:
    # probably running in GitHub actions where there is no data
    # skip all tests in this case
    pytestmark = pytest.mark.skipif(True, reason="Missing MERRA2 data")


@pytest.fixture
def wagga_datetime():
    # simulate date/time for the Wagga example dataset
    # gadi:/g/data/da82/AODH/USGS/L1/Landsat/C1/092_084/LT50920842008269ASA00/...
    #  ... LT05_L1TP_092084_20080925_20161029_01_T1.tar
    return datetime.datetime(2008, 9, 25, 11, 30)


@pytest.fixture
def wagga_lat_long():
    # coords for T intersection in Wagga
    return -35.11559202501883, 147.34547961918634


def _is_valid(aerosol):
    assert -1.0e15 <= aerosol <= 1.0e15  # ensure data within valid range


def test_build_merra2_path(wagga_datetime):
    path = merra2.build_merra2_path(MERRA2_DATA_DIR, wagga_datetime)

    # TODO: temporary until a known MERRA-2 "mirror" is created
    exp = f"{MERRA2_DATA_DIR}/M2T1NXAER.5.12.4/2008/09/MERRA2_300.tavg1_2d_aer_Nx.20080925.nc4"
    assert path == exp


def test_get_closest_value(wagga_datetime, wagga_lat_long):
    path = f"{MERRA2_DATA_DIR}/M2T1NXAER.5.12.4/2008/09/MERRA2_300.tavg1_2d_aer_Nx.20080925.nc4"
    ds = xarray.open_dataset(path)

    aerosol = merra2.get_closest_value(ds, wagga_datetime, wagga_lat_long)
    _is_valid(aerosol)


def test_aerosol_workflow(wagga_datetime, wagga_lat_long):
    coord = (wagga_lat_long,)
    for aerosol in merra2.aerosol_workflow(MERRA2_DATA_DIR, wagga_datetime, coord):
        _is_valid(aerosol)
