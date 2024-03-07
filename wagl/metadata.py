#!/usr/bin/env python

"""Various metadata extraction and creation, and writing tools."""

from datetime import datetime as dtime
from datetime import timezone as dtz

try:
    from importlib.metadata import distribution
except ImportError:
    # Running on pre-3.8 Python; use importlib-metadata package
    from importlib_metadata import distribution

import os
import socket
import uuid
from posixpath import join as ppjoin

import fiona
import h5py
import numpy as np
import pandas as pd
import rasterio
import yaml
from shapely.geometry import Polygon, shape
from yaml.representer import Representer

from wagl.acquisition import Acquisition
from wagl.constants import (
    POINT_FMT,
    BandType,
    BrdfDirectionalParameters,
    BrdfTier,
    DatasetName,
    GroupName,
    Workflow,
)
from wagl.hdf5 import read_h5_table, read_scalar, write_scalar

yaml.add_representer(np.int8, Representer.represent_int)
yaml.add_representer(np.uint8, Representer.represent_int)
yaml.add_representer(np.int16, Representer.represent_int)
yaml.add_representer(np.uint16, Representer.represent_int)
yaml.add_representer(np.int32, Representer.represent_int)
yaml.add_representer(np.uint32, Representer.represent_int)
yaml.add_representer(int, Representer.represent_int)
yaml.add_representer(np.int64, Representer.represent_int)
yaml.add_representer(np.uint64, Representer.represent_int)
yaml.add_representer(float, Representer.represent_float)
yaml.add_representer(np.float32, Representer.represent_float)
yaml.add_representer(np.float64, Representer.represent_float)
yaml.add_representer(np.ndarray, Representer.represent_list)
yaml.add_representer(bool, Representer.represent_bool)


class MetadataError(Exception):
    """Generic error when handling metadata operations."""


def extract_ancillary_metadata(fname):
    """Extracts the change (last metadata change), modified,
    accessed, and owner user id.

    :param fname:
        A string containing the full file pathname to a file
        on disk.

    :return:
        A `dictionary` with keys `ctime`, `mtime`, `atime`,
        and `owner`.
    """

    def _get_utc_datetime(timestamp):
        return dtime.utcfromtimestamp(timestamp).replace(tzinfo=dtz.utc)

    res = {}
    fstat = os.stat(fname)
    res["ctime"] = _get_utc_datetime(fstat.st_ctime)
    res["mtime"] = _get_utc_datetime(fstat.st_mtime)
    res["atime"] = _get_utc_datetime(fstat.st_atime)
    res["owner_id"] = str(fstat.st_uid)
    return res


def get_system_information():
    utc_now = dtime.utcnow().replace(tzinfo=dtz.utc)
    system_info = {
        "uname": " ".join(os.uname()),
        "hostname": socket.getfqdn(),
        "runtime_id": str(uuid.uuid1()),
        "time_processed": utc_now,
    }
    return system_info


def read_metadata_tags(fname, bands):
    """Retrieves the metadata tags for a list of bands from a `GDAL`
    compliant dataset.

    :param fname:
        A string containing the full file pathname to a file
        on disk.

    :param bands:
        A `list` containing the band numbers (1 -> n) from which
        to retreive the metadata tags.

    :return:
        A `pandas.DataFrame`.
    """
    with rasterio.open(fname) as ds:
        tag_data = {k: [] for k in ds.tags(1).keys()}
        for band in bands:
            tags = ds.tags(band)
            for tag in tags:
                tag_data[tag].append(tags[tag])

    return pd.DataFrame(tag_data)


def is_offshore_territory(
    acq: Acquisition, offshore_territory_boundary_path: str
) -> bool:
    geobox = acq.gridded_geo_box()
    acq_polygon = Polygon(
        [geobox.ul_lonlat, geobox.ur_lonlat, geobox.lr_lonlat, geobox.ll_lonlat]
    )

    with fiona.open(offshore_territory_boundary_path, "r") as offshore_territory:
        for boundary_poly in offshore_territory:
            if shape(boundary_poly["geometry"]).contains(acq_polygon):
                return False

    return True


def create_ard_yaml(
    res_group_bands,
    ancillary_group,
    out_group,
    parameters,
    workflow,
    offshore_territory_boundary_path,
):
    """Write the NBAR metadata captured during the entire workflow to a
    HDF5 SCALAR dataset using the yaml document format.

    :param res_group_bands:
        A `dict` mapping resolution group names to lists of `Acquisition` instances.

    :param ancillary_group:
        The root HDF5 `Group` that contains the ancillary data
        collected via wagl.ancillary.collect_ancillary>

    :param out_group:
        A `h5py.Group` object opened for write access.

    :param parameters:
        A `dict` containing `DataStandardisation` parameters

    :param workflow:
        Which workflow to run (from the `wagl.constants.Workflow` enumeration).

    :param offshore_territory_boundary_path:
        GeoJSON to determine if the processed dataset lies outside of
        the best-quality ancillary extent.

    :return:
        None; The yaml document is written to the HDF5 file.
    """
    sbt = workflow in [Workflow.STANDARD, Workflow.SBT]
    nbar = workflow in [Workflow.STANDARD, Workflow.NBAR]

    def load_sbt_ancillary(group):
        """Load the sbt ancillary data retrieved during the worlflow."""
        point_data = {
            DatasetName.DEWPOINT_TEMPERATURE.value: {},
            DatasetName.SURFACE_GEOPOTENTIAL.value: {},
            DatasetName.TEMPERATURE_2M.value: {},
            DatasetName.SURFACE_RELATIVE_HUMIDITY.value: {},
            DatasetName.GEOPOTENTIAL.value: {},
            DatasetName.RELATIVE_HUMIDITY.value: {},
            DatasetName.TEMPERATURE.value: {},
        }

        npoints = group[DatasetName.COORDINATOR.value].shape[0]
        for point in range(npoints):
            pnt_grp = group[POINT_FMT.format(p=point)]
            lonlat = tuple(pnt_grp.attrs["lonlat"])

            # scalars
            dname = DatasetName.DEWPOINT_TEMPERATURE.value
            point_data[dname][lonlat] = read_scalar(pnt_grp, dname)

            dname = DatasetName.SURFACE_GEOPOTENTIAL.value
            point_data[dname][lonlat] = read_scalar(pnt_grp, dname)

            dname = DatasetName.TEMPERATURE_2M.value
            point_data[dname][lonlat] = read_scalar(pnt_grp, dname)

            dname = DatasetName.SURFACE_RELATIVE_HUMIDITY.value
            point_data[dname][lonlat] = read_scalar(pnt_grp, dname)

            # tables
            dname = DatasetName.GEOPOTENTIAL.value
            dset = pnt_grp[dname]
            attrs = dict(dset.attrs.items())
            df = read_h5_table(pnt_grp, dname)
            for column in df.columns:
                attrs[column] = df[column].values
            point_data[dname][lonlat] = attrs

            dname = DatasetName.RELATIVE_HUMIDITY.value
            dset = pnt_grp[dname]
            attrs = dict(dset.attrs.items())
            df = read_h5_table(pnt_grp, dname)
            for column in df.columns:
                attrs[column] = df[column].values
            point_data[dname][lonlat] = attrs

            dname = DatasetName.TEMPERATURE.value
            dset = pnt_grp[dname]
            attrs = dict(dset.attrs.items())
            df = read_h5_table(pnt_grp, dname)
            for column in df.columns:
                attrs[column] = df[column].values
            point_data[dname][lonlat] = attrs

        return point_data

    def load_nbar_ancillary(acquisitions, fid):
        """Load the ancillary data retrieved during the workflow."""
        ids = []
        tier = []
        alphas = {
            "alpha_1": {},
            "alpha_2": {},
        }
        for acq in acquisitions:
            if acq.band_type == BandType.THERMAL:
                continue

            bn = acq.band_name
            for param in BrdfDirectionalParameters:
                fmt = DatasetName.BRDF_FMT.value
                dname = fmt.format(band_name=bn, parameter=param.value)
                dset = fid[dname]
                ids.extend(dset.attrs["id"])
                tier.append(BrdfTier[dset.attrs["tier"]].value)
                alpha_key = param.value.lower().replace("-", "_")
                bn_key = bn.lower().replace("-", "_")
                alphas[alpha_key][bn_key] = dset[()]

        # unique listing of brdf ids
        ids = np.unique(np.array(ids)).tolist()

        # a single tier level will dictate the metadata entry
        tier = BrdfTier(np.min(tier)).name

        result = {
            "id": ids,
            "tier": tier,
            "alpha_1": alphas["alpha_1"],
            "alpha_2": alphas["alpha_2"],
        }

        return result

    def pick_acquisition():
        # pick any acquisition
        band_group = next(iter(res_group_bands))
        return res_group_bands[band_group][0]

    acquisition = pick_acquisition()
    level1_path = acquisition.pathname
    acq_datetime = acquisition.acquisition_datetime.replace(tzinfo=dtz.utc)

    def source_info():
        result = {
            "source_level1": level1_path,
            "acquisition_datetime": acq_datetime,
            "platform_id": acquisition.platform_id,
            "sensor_id": acquisition.sensor_id,
        }
        # ancillary metadata tracking
        result.update(extract_ancillary_metadata(level1_path))
        return result

    def remove_fields(data):
        fields = ["CLASS", "VERSION", "query_date", "data_source"]
        for field in fields:
            data.pop(field, None)
        return data

    def elevation_provenance(anc_grp):
        ids = []

        # low resolution source
        dname = DatasetName.ELEVATION.value
        dset = anc_grp[dname]
        ids.extend(dset.attrs["id"])

        # high resolution source (res group is adjacent to ancillary group)
        parent_group = anc_grp.parent
        for res_group in res_group_bands:
            dname = ppjoin(
                res_group,
                GroupName.ELEVATION_GROUP.value,
                DatasetName.DSM_SMOOTHED.value,
            )
            dset = parent_group[dname]
            ids.extend(dset.attrs["id"])

        # unique listing of ids
        ids = np.unique(np.array(ids)).tolist()
        md = {
            "id": ids,
        }

        return md

    def ancillary(fid):
        # load the ancillary and remove fields not of use to ODC
        # retrieve the averaged ancillary if available
        anc_grp = fid.get(GroupName.ANCILLARY_AVG_GROUP.value)
        if anc_grp is None:
            anc_grp = fid

        dname = DatasetName.AEROSOL.value
        aerosol_data = remove_fields(read_scalar(anc_grp, dname))
        dname = DatasetName.WATER_VAPOUR.value
        water_vapour_data = remove_fields(read_scalar(anc_grp, dname))
        dname = DatasetName.OZONE.value
        ozone_data = remove_fields(read_scalar(anc_grp, dname))

        # currently have multiple sources of elevation data
        elevation_data = elevation_provenance(anc_grp)

        result = {
            "aerosol": aerosol_data,
            "water_vapour": water_vapour_data,
            "ozone": ozone_data,
            "elevation": elevation_data,
        }

        if sbt:
            result.update(load_sbt_ancillary(fid))

        if nbar:
            result["brdf"] = {}
            for grp_name in res_group_bands:
                grp_ancillary = load_nbar_ancillary(res_group_bands[grp_name], fid)
                if result["brdf"] == {}:
                    result["brdf"] = grp_ancillary
                else:
                    brdf = result["brdf"]
                    brdf["id"] = list(set(brdf["id"]) | set(grp_ancillary["id"]))
                    brdf["tier"] = BrdfTier(
                        min(
                            BrdfTier[brdf["tier"]].value,
                            BrdfTier[grp_ancillary["tier"]].value,
                        )
                    ).name
                    brdf["alpha_1"].update(grp_ancillary["alpha_1"])
                    brdf["alpha_2"].update(grp_ancillary["alpha_2"])

        return result

    def software_versions():
        dist = distribution("ard-pipeline")
        return {
            "wagl": {
                "version": dist.version,
                "repo_url": dist.metadata.get("Home-page"),
            },
            "modtran": {
                "version": "6.0.1",
                "repo_url": "http://www.ontar.com/software/productdetails.aspx?item=modtran",
            },
        }

    def algorithm():
        result = {}

        if sbt:
            result["sbt_doi"] = "TODO"

        if nbar:
            result["algorithm_version"] = 2.0
            result["nbar_doi"] = "http://dx.doi.org/10.1109/JSTARS.2010.2042281"
            result["nbar_terrain_corrected_doi"] = (
                "http://dx.doi.org/10.1016/j.rse.2012.06.018"
            )

        return result

    metadata = {
        "system_information": get_system_information(),
        "source_datasets": source_info(),
        "ancillary": ancillary(ancillary_group),
        "algorithm_information": algorithm(),
        "software_versions": software_versions(),
        "id": str(uuid.uuid4()),
        "parameters": parameters,
    }

    offshore = is_offshore_territory(acquisition, offshore_territory_boundary_path)
    if offshore:
        metadata["final_ancillaries"] = "nonstandard"

    # output
    yml_data = yaml.dump(metadata, default_flow_style=False)
    write_scalar(yml_data, metadata["id"], out_group, attrs={"file_format": "yaml"})
    out_group[DatasetName.CURRENT_METADATA.value] = h5py.SoftLink(
        "{}/{}".format(out_group.name, metadata["id"])
    )


def current_h5_metadata(fid: h5py.Group, dataset_path: str = ""):
    """Read metadata entrypoint from h5 collection.

    :param fid:
        A h5py.Group that includes the dataset metadata

    :param dataset_path:
        An optional reference (string) to the dataset location

    :raises:
        :MetadataError: Returned when a metadata document couldn't be found for
            dataset provided

    :return:
        A dictionary representation of the dataset metadata
    """
    metadata = fid.get(
        "/{}/{}/{}".format(
            DatasetName.METADATA.value,
            dataset_path.lstrip("/"),
            DatasetName.CURRENT_METADATA.value,
        )
    )

    if not metadata:  # assume h5 collection represents 1 dataset
        metadata = fid.get(
            f"/{DatasetName.METADATA.value}/{DatasetName.CURRENT_METADATA.value}"
        )
        if not metadata:
            raise MetadataError(
                f"Unable to find metadata entry for dataset: {fid.filename}:{dataset_path}"
            )

    return yaml.load(metadata[()].item(), Loader=yaml.FullLoader)
