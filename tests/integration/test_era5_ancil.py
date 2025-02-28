import os
import socket

import h5py
import pytest

from wagl import acquisition, ancillary, constants

# check TMPDIR for testing
TMP_DIR = "TMPDIR"

if TMP_DIR not in os.environ:
    # raise RuntimeError("Set $TMPDIR environment var for era5 integration testing")
    TMP_DIR = False


@pytest.fixture
def nci_era5_dir_path():
    # HACK: this is NCI platform specific
    if "gadi" in socket.gethostname():
        return "/g/data/rt52/era5"

    msg = "This test is currently coupled to the NCI environment"
    raise RuntimeError(msg)


@pytest.fixture
def cop_sentinel2_root():
    return "/g/data/fj7/Copernicus/Sentinel-2/MSI/L1C/"


@pytest.fixture
def canberra_scene_sentinel2_path(cop_sentinel2_root):
    # NB: return NCI specific path to test scene over Aus
    path = "2024/2024-12/30S145E-35S150E/S2A_MSIL1C_20241207T001111_N0511_R073_T55HDD_20241207T011213.zip"
    abs_path = os.path.join(cop_sentinel2_root, path)
    assert os.path.exists(abs_path)
    return abs_path


@pytest.fixture
def canberra_scene_sentinel2_container(canberra_scene_sentinel2_path):
    container = acquisition.acquisitions(canberra_scene_sentinel2_path)
    return container


@pytest.fixture
def scene_landsat_path():
    gdata = "/g/data/da82/AODH/USGS/L1/Landsat"
    p = "/C1/092_084/LT50920842008269ASA00/LT05_L1TP_092084_20080925_20161029_01_T1.tar"
    path = os.path.join(gdata, p)
    assert os.path.exists(path)
    return path


@pytest.fixture
def scene_landsat_container(scene_landsat_path):
    container = acquisition.acquisitions(scene_landsat_path)
    return container


@pytest.fixture
def output_filename(canberra_scene_sentinel2_path):
    bn = os.path.basename(canberra_scene_sentinel2_path)
    no_ext = os.path.splitext(bn)[0]
    out = f"{no_ext}.testing.wagl.h5"
    return out


@pytest.mark.skipif(TMP_DIR is False, reason="Test system is not NCI")
def test_collect_era5_ancillary(
    scene_landsat_container, nci_era5_dir_path, output_filename
):
    # config copied from luigi cfg template & singlefile_workflow.py
    cfg_paths = {
        "aerosol": 0.05,
        "dem_path": "/g/data/v10/eoancillarydata-2/elevation/world_1deg/DEM_one_deg_20June2019.h5:/SRTM/GA-DSM",
        "brdf_dict": {
            "brdf_path": "/g/data/v10/eoancillarydata-2/BRDF/MCD43A1.061",
            "ocean_mask_path": "/g/data/v10/eoancillarydata-2/ocean_mask/base_oz_tile_set_water_mask_geotif.tif",
        },
    }

    tmp_dir = os.path.abspath(os.environ[TMP_DIR])
    tmp_dir = os.path.join(tmp_dir, "ard-era5-testing")

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    dest = os.path.join(tmp_dir, output_filename)

    with h5py.File(dest, "w") as fid:
        out_group = fid.create_group(constants.GroupName.ANCILLARY_GROUP.value)

        ancillary.collect_era5_ancillary(
            scene_landsat_container, nci_era5_dir_path, cfg_paths, out_group
        )

    assert os.path.exists(dest)
