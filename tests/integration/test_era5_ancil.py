"""
ERA5 Ancillary Integration testing.

This is a quick & dirty integration test script, designed to run blocks of the
ERA5 workflow in isolation from the full `ard-pipeline` workflow. It's treated
as an integration test as it executes multiple python modules & reads real data
for testing.

NB: this code is **coupled to NCI** systems due to data access requirements.

NCI usage:

This test script is intended to be executed from an `ard-pipeline` conda
environment. If you do not have conda environment, set one up using the README.

1) Log into `gadi`

2) Activate the `ard-pipeline` environment.

3) `cd` to your git clone of `ard-pipeline`.

4) Run `pytest --disable-warnings tests/integration/test_era5_ancil.py`

Test runs may take anywhere from 1 to 10 minutes. The cause is unknown as NCI
nodes have non-deterministic performance issues.
"""

import os
import socket

import h5py
import pytest

from wagl import acquisition, ancillary, constants

# check hostname & TMPDIR for platform checking
if "gadi" not in socket.gethostname():
    TMP_DIR = False
else:
    TMP_DIR = "TMPDIR"

    if TMP_DIR not in os.environ:
        msg = "Set $TMPDIR env variable to temp directory"
        raise RuntimeError(msg)

_REASON = "Platform does not appear to be an NCI system"


@pytest.fixture
def nci_era5_dir_path():
    # HACK: this is NCI platform specific
    if "gadi" in socket.gethostname():
        return "/g/data/rt52/era5"

    msg = "This test is currently coupled to the NCI environment"
    raise RuntimeError(msg)


@pytest.fixture
def cop_sentinel2_root():
    # shorten long NCI file paths with this fixture
    return "/g/data/fj7/Copernicus/Sentinel-2/MSI/L1C/"


@pytest.fixture
def canberra_scene_sentinel2_path(cop_sentinel2_root):
    # return NCI specific path to test scene over Aus, use 2023 to ensure ERA5
    # data exists (as some 2024 data was missing as of 03/2025)
    p = "2023/2023-12/35S145E-40S150E/S2B_MSIL1C_20231231T235229_N0510_R130_T55HGS_20240101T004900.zip"
    abs_path = os.path.join(cop_sentinel2_root, p)
    assert os.path.exists(abs_path)
    return abs_path


@pytest.fixture
def canberra_scene_sentinel2_container(canberra_scene_sentinel2_path):
    container = acquisition.acquisitions(canberra_scene_sentinel2_path)
    return container


@pytest.fixture
def scene_landsat_path():
    gdata = "/g/data/da82/AODH/USGS/L1/Landsat"
    p = "C1/092_084/LT50920842008269ASA00/LT05_L1TP_092084_20080925_20161029_01_T1.tar"
    path = os.path.join(gdata, p)
    assert os.path.exists(path)
    return path


@pytest.fixture
def scene_landsat_container(scene_landsat_path):
    container = acquisition.acquisitions(scene_landsat_path)
    return container


@pytest.fixture
def output_filename_sentinel(canberra_scene_sentinel2_path):
    bn = os.path.basename(canberra_scene_sentinel2_path)
    no_ext = os.path.splitext(bn)[0]
    out = f"{no_ext}.testing.wagl.h5"
    return out


@pytest.fixture
def output_filename_landsat(scene_landsat_path):
    bn = os.path.basename(scene_landsat_path)
    no_ext = os.path.splitext(bn)[0]
    out = f"{no_ext}.testing.wagl.h5"
    return out


# config copied from luigi cfg template & singlefile_workflow.py
_default_cfg_paths = {
    "aerosol": 0.05,
    "dem_path": "/g/data/v10/eoancillarydata-2/elevation/world_1deg/DEM_one_deg_20June2019.h5:/SRTM/GA-DSM",
    "brdf_dict": {
        "brdf_path": "/g/data/v10/eoancillarydata-2/BRDF/MCD43A1.061",
        "ocean_mask_path": "/g/data/v10/eoancillarydata-2/ocean_mask/base_oz_tile_set_water_mask_geotif.tif",
    },
}


def init_tmp_dir():
    tmp_dir = os.path.abspath(os.environ[TMP_DIR])
    tmp_dir = os.path.join(tmp_dir, "ard-era5-testing")

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    return tmp_dir


@pytest.mark.skipif(TMP_DIR is False, reason=_REASON)
def test_collect_era5_ancillary_landsat(
    scene_landsat_container, nci_era5_dir_path, output_filename_landsat
):
    tmp_dir = init_tmp_dir()
    dest_path = os.path.join(tmp_dir, output_filename_landsat)

    with h5py.File(dest_path, "w") as fid:
        out_group = fid.create_group(constants.GroupName.ANCILLARY_GROUP.value)

        ancillary.collect_era5_ancillary(
            scene_landsat_container, nci_era5_dir_path, _default_cfg_paths, out_group
        )

    assert os.path.exists(dest_path)

    # verify H5 file contains ancillary data
    # NB: as of 03/2025, ard-pipeline xarray version is 2024.02.0. This version
    #  lacks features & can't seem to read H5 subgroups like POINT-0. Use h5py
    #  for testing for now
    df = h5py.File(dest_path)

    expected_aerosol = 0.05
    assert df["ANCILLARY/AEROSOL"][()] == expected_aerosol

    # NB: expected_ozone = "TODO"
    ozone = df["ANCILLARY/OZONE"][()]
    assert ozone is not None
    assert ozone != 0.0  # FIXME: copy ozone from source data?

    # NB: expected elevation = ???
    elevation = df["ANCILLARY/ELEVATION"][()]
    assert elevation is not None
    assert elevation > 0.0  # FIXME: copy elevation from source?

    # check atmos profile
    profile = df["ANCILLARY/POINT-0/ATMOSPHERIC-PROFILE"]
    assert profile is not None
    assert len(profile) == 38  # number of rows
    p0 = tuple(profile[0])  # convert ndarray to tuple to allow slicing
    assert p0[0] == 0  # check 1st row index is valid

    for val in p0[1:]:
        assert val is not None
        assert val != 0

    plast = tuple(profile[-1])
    assert plast[0] == 37  # check valid last row index

    for val in plast[1:]:
        assert val is not None
        assert val != 0


@pytest.mark.skipif(TMP_DIR is False, reason=_REASON)
def test_collect_era5_ancillary_sentinel(
    canberra_scene_sentinel2_container, nci_era5_dir_path, output_filename_sentinel
):
    tmp_dir = init_tmp_dir()
    dest_path = os.path.join(tmp_dir, output_filename_sentinel)

    with h5py.File(dest_path, "w") as fid:
        out_group = fid.create_group(constants.GroupName.ANCILLARY_GROUP.value)

        ancillary.collect_era5_ancillary(
            canberra_scene_sentinel2_container,
            nci_era5_dir_path,
            _default_cfg_paths,
            out_group,
        )

    assert os.path.exists(dest_path)

    # very basic test to ensure readable HDF5 output...
    df = h5py.File(dest_path)

    expected_aerosol = 0.05
    assert df["ANCILLARY/AEROSOL"][()] == expected_aerosol
