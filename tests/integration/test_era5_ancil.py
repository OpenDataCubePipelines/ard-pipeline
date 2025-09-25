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
import warnings

import h5py
import pytest

from wagl import acquisition, ancillary, constants

# check hostname & TMPDIR for platform checking
# this could provide false matches for hostnames containing "gadi"
on_gadi = "gadi" in socket.gethostname()

if on_gadi:
    TMP_DIR = "TMPDIR"
else:
    msg = "Running era5 ancil testing not supported outside NCI yet..."
    warnings.warn(msg)

if on_gadi and TMP_DIR not in os.environ:
    msg = "Set $TMPDIR env variable to temp directory"
    raise RuntimeError(msg)

_REASON = "Platform unrecognised as NCI system"


@pytest.fixture
def nci_era5_dir_path():
    # HACK: this module is NCI platform specific
    if on_gadi:
        return "/g/data/rt52/era5"

    msg = "This fixture is currently coupled to the NCI environment"
    raise RuntimeError(msg)


@pytest.fixture
def sentinel2_root():
    # shorten long NCI file paths with this fixture
    return "/g/data/fj7/Copernicus/Sentinel-2/MSI/L1C/"


@pytest.fixture
def wagga_scene_sentinel2_path(sentinel2_root):
    # return NCI specific path to test scene over Aus, use 2023 to ensure ERA5
    # data exists (as some 2024 data was missing as of 03/2025)
    p = "2021/2021-01/30S145E-35S150E/S2B_MSIL1C_20210122T001109_N0209_R073_T55HEB_20210122T012405.zip"
    abs_path = os.path.join(sentinel2_root, p)
    assert os.path.exists(abs_path)
    return abs_path


@pytest.fixture
def wagga_scene_sentinel2_container(wagga_scene_sentinel2_path):
    container = acquisition.acquisitions(wagga_scene_sentinel2_path)
    return container


@pytest.fixture
def scene_landsat_path():
    gdata = "/g/data/da82/AODH/USGS/L1/Landsat"
    p = "C1/092_084/LT50920842008269ASA00/LT05_L1TP_092084_20080925_20161029_01_T1.tar"
    path = os.path.join(gdata, p)
    assert os.path.exists(path)
    return path


@pytest.fixture
def scene_landsat_base_path(scene_landsat_path):
    """Return Landsat directory name."""
    # HACK: parsing the fixture saves copying a substring should test data change
    return scene_landsat_path.split("/")[-2]


@pytest.fixture
def scene_landsat_container(scene_landsat_path):
    container = acquisition.acquisitions(scene_landsat_path)
    return container


@pytest.fixture
def output_filename_sentinel(wagga_scene_sentinel2_path):
    bn = os.path.basename(wagga_scene_sentinel2_path)
    no_ext = os.path.splitext(bn)[0]
    out = f"{no_ext}.testing.wagl.h5"
    return out


@pytest.fixture
def output_filename_landsat(scene_landsat_path):
    bn = os.path.basename(scene_landsat_path)
    no_ext = os.path.splitext(bn)[0]
    out = f"{no_ext}.testing.wagl.h5"
    return out


def init_tmp_dir():
    tmp_dir = os.path.abspath(os.environ[TMP_DIR])
    tmp_dir = os.path.join(tmp_dir, "ard-era5-testing")

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    return tmp_dir


@pytest.mark.skipif(not on_gadi, reason=_REASON)
def test_collect_era5_ancillary_landsat(
    scene_landsat_container,
    scene_landsat_base_path,
    nci_era5_dir_path,
    output_filename_landsat,
):
    tmp_dir = init_tmp_dir()
    dest_path = os.path.join(tmp_dir, output_filename_landsat)

    with h5py.File(dest_path, "w") as fid:
        root_group = fid.create_group(scene_landsat_base_path)
        out_group = root_group.create_group(constants.GroupName.ANCILLARY_GROUP.value)
        centroid = [(-34.62198174915786, 146.75891632807912)]

        ancillary.collect_era5_ancillary(
            scene_landsat_container.get_highest_resolution()[0][0],
            centroid,
            nci_era5_dir_path,
            out_group,
        )

    assert os.path.exists(dest_path)

    # verify H5 file contains ancillary data
    # NB: as of 03/2025, ard-pipeline xarray version is 2024.02.0. This version
    #  lacks features & can't seem to read H5 subgroups like POINT-0. Use h5py
    #  for testing for now
    df = h5py.File(dest_path)
    base = scene_landsat_base_path

    # check atmospheric profile
    profile = df[f"{base}/ANCILLARY/POINT-0/ATMOSPHERIC-PROFILE"]
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


@pytest.mark.skipif(not on_gadi, reason=_REASON)
def test_collect_era5_ancillary_landsat_multi_points(
    scene_landsat_container,
    scene_landsat_base_path,
    nci_era5_dir_path,
    output_filename_landsat,
):
    # direct to alt file to prevent I/O clash
    tmp_dir = init_tmp_dir()
    dest_path = os.path.join(tmp_dir, "multi_point_" + output_filename_landsat)

    with h5py.File(dest_path, "w") as fid:
        root_group = fid.create_group(scene_landsat_base_path)
        out_group = root_group.create_group(constants.GroupName.ANCILLARY_GROUP.value)

        lat_longs = [
            (-34.62198174915786, 146.75891632807912),  # centroid
            (-35.11559202501883, 147.34547961918634),  # Wagga T intersection
        ]

        ancillary.collect_era5_ancillary(
            scene_landsat_container.get_highest_resolution()[0][0],
            lat_longs,
            nci_era5_dir_path,
            out_group,
        )

    # check multi point atmos profile exists
    df = h5py.File(dest_path)
    base = scene_landsat_base_path

    profile = df[f"{base}/ANCILLARY/POINT-0/ATMOSPHERIC-PROFILE"]
    assert profile is not None  # same as profile 1 in the single location test

    profile2 = df[f"{base}/ANCILLARY/POINT-1/ATMOSPHERIC-PROFILE"]
    assert profile2 is not None

    assert len(profile2) in [37, 38]  # num rows, account for possible profile inversion
    p0 = tuple(profile2[0])  # convert ndarray to tuple to allow slicing
    assert p0[0] == 0  # check 1st row index is valid

    for val in p0[1:]:
        assert val is not None
        assert val != 0

    plast = tuple(profile2[-1])
    assert plast[0] == 37  # check valid last row index

    for val in plast[1:]:
        assert val is not None
        assert val != 0


@pytest.mark.skipif(not on_gadi, reason=_REASON)
def test_collect_era5_ancillary_sentinel(
    wagga_scene_sentinel2_container, nci_era5_dir_path, output_filename_sentinel
):
    tmp_dir = init_tmp_dir()
    dest_path = os.path.join(tmp_dir, output_filename_sentinel)

    acq = wagga_scene_sentinel2_container.get_highest_resolution()[0][0]
    geobox = acq.gridded_geo_box()
    latlongs = (geobox.centre_lonlat[::-1], (-35.11559202501883, 147.34547961918634))

    # root group name copies naming from workflow H5 output files
    rootname = wagga_scene_sentinel2_container.granules[0]

    with h5py.File(dest_path, "w") as fid:
        root_group = fid.create_group(rootname)  # mimic scene_landsat_base_path
        out_group = root_group.create_group(constants.GroupName.ANCILLARY_GROUP.value)

        ancillary.collect_era5_ancillary(
            wagga_scene_sentinel2_container.get_highest_resolution()[0][0],
            latlongs,
            nci_era5_dir_path,
            out_group,
        )

    assert os.path.exists(dest_path)

    # very basic test to ensure readable HDF5 output with new ancillaries...
    df = h5py.File(dest_path)

    profile = df[f"{rootname}/ANCILLARY/POINT-0/ATMOSPHERIC-PROFILE"]
    assert profile is not None  # same as profile 1 in the single location test

    profile2 = df[f"{rootname}/ANCILLARY/POINT-1/ATMOSPHERIC-PROFILE"]
    assert profile2 is not None
    assert len(profile2) in [37, 38]  # num rows, account for possible profile inversion
