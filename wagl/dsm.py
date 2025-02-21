#!/usr/bin/env python
"""Digital Surface Model Data extraction and smoothing."""

import itertools
import os.path
from math import ceil, degrees, floor, radians

import boto3
import h5py
import numpy as np
import rasterio
from botocore import UNSIGNED
from botocore.config import Config
from osgeo import osr
from rasterio.io import MemoryFile
from rasterio.warp import Resampling, reproject
from scipy import ndimage
from sklearn.metrics.pairwise import haversine_distances

from wagl.constants import DatasetName, GroupName
from wagl.data import read_subset, reproject_array_to_array
from wagl.geobox import GriddedGeoBox
from wagl.hdf5 import VLEN_STRING, H5CompressionFilter, attach_image_attributes
from wagl.margins import pixel_buffer
from wagl.metadata import current_h5_metadata


def filter_dsm(array):
    """Applies a gaussian filter to array.

    :param array:
        A 2D NumPy array.

    :return:
        A 2D NumPy array.
    """
    # Define the kernel
    kernel = [
        0.009511,
        0.078501,
        0.009511,
        0.078501,
        0.647954,
        0.078501,
        0.009511,
        0.078501,
        0.009511,
    ]
    kernel = np.array(kernel).reshape((3, 3))

    filtered = ndimage.convolve(array, kernel)
    return filtered


def read_subset_to_geobox(dsm_dataset, dem_geobox):
    dsm_geobox = GriddedGeoBox.from_dataset(dsm_dataset)

    # calculate full border extents into CRS of DSM
    extents = dem_geobox.project_extents(dsm_geobox.crs)
    ul_xy = (extents[0], extents[3])
    ur_xy = (extents[2], extents[3])
    lr_xy = (extents[2], extents[1])
    ll_xy = (extents[0], extents[1])

    # load the subset and corresponding geobox
    subs, subs_geobox = read_subset(
        dsm_dataset, ul_xy, ur_xy, lr_xy, ll_xy, edge_buffer=1
    )

    # Retrieve the DSM data
    dsm_data = reproject_array_to_array(
        subs, subs_geobox, dem_geobox, resampling=Resampling.bilinear
    )

    # free memory
    subs = None

    return dsm_data


def copernicus_tiles_latlon_covering_extents(lat_lon_extents):
    """
    Yields (lat, long) tuples of CopDEM tile coordinates.

    These coordinates correspond to the CopDEM tile numbering scheme & extents
    for each tile. Tile extents vary by hemisphere.

    e.g. partial `gdalinfo` output for Copernicus_DSM_COG_10_S34_00_E148_00_DEM.tif:
    ...
    Upper Left  ( 147.9998611, -32.9998611) --> (148, -33)
    Lower Left  ( 147.9998611, -33.9998611) --> (148, -34)
    Upper Right ( 148.9998611, -32.9998611) --> (149, -33)
    Lower Right ( 148.9998611, -33.9998611) --> (149, -34)
    ...
    In the southern hemisphere, S34 tile latitude extents are -33 to -34 degrees.
    The S34 tile includes the extent to 34 degrees.

    For a northern hemisphere tile Copernicus_DSM_COG_10_N34_00_E125_00_DEM.tif:
    ...
    Upper Left  ( 124.9998611,  35.0001389) --> (125, 35)
    Lower Left  ( 124.9998611,  34.0001389) --> (125, 34)
    Upper Right ( 125.9998611,  35.0001389) --> (126, 35)
    Lower Right ( 125.9998611,  34.0001389) --> (126, 34)
    ...
    The N34 tile latitude extents are 34 to 35 degrees. It's slightly different
    to the southern hemisphere with the N34 tile extents >= 34 degrees.

    :param lat_lon_extents: (min_x, min_y, max_x, max_y) tuple
    """
    # Convert floating point extents to integer coords for CopDEM tile numbering
    #
    # Tile naming depends on the latitude/longitude floor operation. A latitude
    # coordinate extent of N34.5 has floor(34.5) --> 34, mapping to the N34 tile
    # with its extent covering 34 to 35 degrees.
    #
    # "Negative" coordinates are more subtle. A latitude coord extent of S33.5
    # has floor(-33.5) --> -34, mapping to tile S34 (-33 to -34 degrees).
    from_lon, from_lat, to_lon, to_lat = (floor(n) for n in lat_lon_extents)

    def order(a, b):
        return (a, b) if a <= b else (b, a)

    from_lat, to_lat = order(from_lat, to_lat)
    from_lon, to_lon = order(from_lon, to_lon)

    if disjoint_antimeridian(lat_lon_extents):
        yield from itertools.product(
            range(from_lat, to_lat + 1), range(from_lon, to_lon + 1)
        )
    else:
        # Antimeridian handling
        # For longitudes, emit coordinates starting at antimeridian, traversing
        # west to east across *western* hemisphere coordinates. Emit smallest
        # longitudes 1st, equivalent to sliding right across the scene
        yield from itertools.product(
            range(from_lat, to_lat + 1), range(-180, from_lon + 1)
        )

        # Continue emitting longitudes from the scene's westernmost extent, which
        # occupies eastern hemisphere coordinate space (also slide right)
        yield from itertools.product(range(from_lat, to_lat + 1), range(to_lon, 180))


def disjoint_antimeridian(lat_lon_extents):
    """
    Return True if the extents do not cross the antimeridian.
    """
    # Inverted intersects_antimeridian() for simpler logic in calling functions
    return not intersects_antimeridian(lat_lon_extents)


def intersects_antimeridian(lat_lon_extents):
    """
    Return True if the longitude extents cross the +/- 180 antimeridian
    """
    from_lon, _, to_lon, to_lat = lat_lon_extents
    to_lat_r = radians(to_lat)  # NB: use 1 lat to focus on longitude diff only
    from_lon_r, to_lon_r = tuple(radians(n) for n in sorted((from_lon, to_lon)))

    p0 = (to_lat_r, from_lon_r)  # NB: using 1 latitude for horizontal distance
    p1 = (to_lat_r, to_lon_r)

    # Use haversine distance of scene longitude extent points to calculate the
    # scene width (the delta). Use delta distance to determine if this arc
    # between points crosses the antimeridian.
    #
    # haversine_distances() returns 2x2 arrays in radians:
    # array([[0.        , 0.04654447],
    #        [0.04654447, 0.        ]])
    delta_radians = haversine_distances((p0, p1))
    lon_delta_radians = np.unique(delta_radians[delta_radians > 0.0])
    lon_delta_degrees = degrees(lon_delta_radians)

    # 'extend' a line from min longitude for antimeridian check
    return abs(from_lon) + lon_delta_degrees > 180.0


def copernicus_folder_for_latlon(lat, lon) -> str:
    lat_str = f"N{abs(lat):02d}" if lat >= 0 else f"S{abs(lat):02d}"
    lon_str = f"E{abs(lon):03d}" if lon >= 0 else f"W{abs(lon):03d}"
    return f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM"


def copernicus_dem_image_for_latlon(lat, lon) -> str:
    key_id = copernicus_folder_for_latlon(lat, lon)
    return f"{key_id}/{key_id}.tif"


def copernicus_wbm_image_for_latlon(lat, lon) -> str:
    key_id = copernicus_folder_for_latlon(lat, lon)
    wbm_id = key_id.replace("_DEM", "_WBM")
    return f"{key_id}/AUXFILES/{wbm_id}.tif"


def split_s3_path_into_bucket_and_prefix(s3_path: str) -> tuple[str, str]:
    assert s3_path.startswith("s3://")
    path = s3_path[len("s3://") :]
    bucket = path.split("/")[0]
    prefix = path[len(bucket) :]
    if prefix.startswith("/"):
        prefix = prefix[1:]
    return bucket, prefix


def read_s3_object_into_memory(bucket, key):
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    try:
        buffer = MemoryFile(filename=os.path.basename(key))
        s3.download_fileobj(bucket, key, buffer)
        return buffer
    except s3.exceptions.NoSuchKey:
        raise FileNotFoundError(f"Failed to get cop30 DEM tile for s3://{bucket}/{key}")


def covering_geobox_subset(dst_geobox, src_geobox):
    """
    Calculate the sub-set (as slice objects of indices) of `dst_geobox` that covers
    its intersection with `src_geobox` so that the contents of `src_geobox`
    can be read into that subset.
    """
    extents = src_geobox.project_extents(dst_geobox.crs)

    # p1 and p2 are in "pixel space"
    p1 = ~dst_geobox.transform * (extents[0], extents[3])
    p2 = ~dst_geobox.transform * (extents[2], extents[1])

    shape = dst_geobox.shape
    minx = max(floor(min(p1[0], p2[0])), 0)
    miny = max(floor(min(p1[1], p2[1])), 0)
    maxx = min(ceil(max(p1[0], p2[0])), shape[1])
    maxy = min(ceil(max(p1[1], p2[1])), shape[0])

    if (minx >= maxx) or (miny >= maxy):
        return None, None

    subset_geobox = GriddedGeoBox(
        (maxy - miny, maxx - minx),
        origin=dst_geobox.transform * (minx, miny),
        pixelsize=dst_geobox.pixelsize,
        crs=dst_geobox.crs.ExportToProj4(),
    )
    return (slice(miny, maxy), slice(minx, maxx)), subset_geobox


def list_copernicus_bands_for_geobox(cop_pathname, key_to_path, dst_geobox):
    """
    `key_to_path` is a function that takes a folder name and translates
    it to a band path.
    """
    if not os.path.isdir(cop_pathname) and not cop_pathname.startswith("s3://"):
        raise ValueError("Not a valid tiled Copernicus DEM")

    # is it on disk or in a s3 bucket
    cached = os.path.isdir(cop_pathname)

    if cached:
        bucket, prefix = None, cop_pathname
    else:
        bucket, prefix = split_s3_path_into_bucket_and_prefix(cop_pathname)

    if prefix != "" and not prefix.endswith("/"):
        prefix = prefix + "/"

    cop30m_crs = osr.SpatialReference()
    cop30m_crs.ImportFromEPSG(4326)  # WGS84
    lat_lon_extents = dst_geobox.project_extents(cop30m_crs)

    for lat, lon in copernicus_tiles_latlon_covering_extents(lat_lon_extents):
        location = prefix + key_to_path(lat, lon)

        if cached:
            if not os.path.isfile(location):
                # ocean tiles are not present
                continue

            dataset_reader = rasterio.open(location)
        else:
            try:
                dataset_reader = read_s3_object_into_memory(bucket, location).open()
            except FileNotFoundError:
                continue

        yield dataset_reader


def read_copernicus_dem(cop_pathname, dst_geobox):
    result = np.full(dst_geobox.shape, np.nan, dtype=np.float32)
    dst_crs = dst_geobox.crs.ExportToProj4()

    for dataset_reader in list_copernicus_bands_for_geobox(
        cop_pathname, copernicus_dem_image_for_latlon, dst_geobox
    ):
        with dataset_reader as ds:
            subset, subset_geobox = covering_geobox_subset(
                dst_geobox, GriddedGeoBox.from_dataset(ds)
            )

            if subset is None:
                continue

            try:
                reprojected = np.full(subset_geobox.shape, np.nan, dtype=np.float32)

                if not (reprojected.shape[0] > 0 and reprojected.shape[1] > 0):
                    # otherwise reproject below fails
                    raise ValueError

                reproject(
                    source=rasterio.band(ds, 1),
                    destination=reprojected,
                    dst_crs=dst_crs,
                    dst_transform=subset_geobox.transform,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
                result[subset] = np.where(
                    np.isnan(result[subset]), reprojected, result[subset]
                )
            except ValueError:
                # TODO investigate this
                # possibly DEM tile not intersecting with the dst_geobox
                pass

    result = np.where(np.isnan(result), 0.0, result)
    return result


def get_dsm(
    acquisition,
    srtm_pathname,
    cop_pathname,
    buffer_distance=15000,
    out_group=None,
    compression=H5CompressionFilter.LZF,
    filter_opts=None,
):
    """Given an acquisition and a national Digital Surface Model,
    extract a subset from the DSM based on the acquisition extents
    plus an x & y margins. The subset is then smoothed with a 3x3
    gaussian filter.
    A square margin is applied to the extents.

    :param acquisition:
        An instance of an acquisition object.

    :param srtm_pathname:
        A string pathname of the SRTM DSM with a ':' to separate the
        filename from the import HDF5 dataset name.

    :param cop_pathname:
        A string pathname of the mosaiced Copernicus 30m DEM .tif file.
        Alternatively, a folder containing the Copernicus 30m DEM, or
        an S3 location for the Copernicus 30m DEM.

    :param buffer_distance:
        A number representing the desired distance (in the same
        units as the acquisition) in which to calculate the extra
        number of pixels required to buffer an image.
        Default is 15000.

    :param out_group:
        A writeable HDF5 `Group` object.

        The dataset name will be as follows:

        * DatasetName.DSM_SMOOTHED

    :param compression:
        The compression filter to use.
        Default is H5CompressionFilter.LZF

    :param filter_opts:
        A dict of key value pairs available to the given configuration
        instance of H5CompressionFilter. For example
        H5CompressionFilter.LZF has the keywords *chunks* and *shuffle*
        available.
        Default is None, which will use the default settings for the
        chosen H5CompressionFilter instance.

    :return:
        An opened `h5py.File` object, that is either in-memory using the
        `core` driver, or on disk.
    """
    # Use the 1st acquisition to set up the geobox
    geobox = acquisition.gridded_geo_box()
    shape = geobox.get_shape_yx()

    # buffered image extents/margins
    margins = pixel_buffer(acquisition, buffer_distance)

    # Get the dimensions and geobox of the new image
    dem_cols = shape[1] + margins.left + margins.right
    dem_rows = shape[0] + margins.top + margins.bottom
    dem_shape = (dem_rows, dem_cols)
    dem_origin = geobox.convert_coordinates((0 - margins.left, 0 - margins.top))

    dem_geobox = GriddedGeoBox(
        dem_shape,
        origin=dem_origin,
        pixelsize=geobox.pixelsize,
        crs=geobox.crs.ExportToWkt(),
    )

    try:
        # split the DSM filename, dataset name, and load
        fname, dname = srtm_pathname.split(":")
        with h5py.File(fname, "r") as dsm_fid:
            dsm_ds = dsm_fid[dname]
            dsm_data = read_subset_to_geobox(dsm_ds, dem_geobox)

            # ancillary metadata tracking
            metadata = current_h5_metadata(dsm_fid, dataset_path=dname)

    except (IndexError, ValueError):
        # ancillary metadata tracking
        metadata = {"id": "cop-30m-dem"}

        if os.path.isfile(cop_pathname):
            # read from mosaic
            with rasterio.open(cop_pathname, "r") as dsm_ds:
                dsm_data = read_subset_to_geobox(dsm_ds, dem_geobox)
        elif cop_pathname.startswith("s3://") or os.path.isdir(cop_pathname):
            # read from tiled CopDEM30m
            dsm_data = read_copernicus_dem(cop_pathname, dem_geobox)
        else:
            raise ValueError("No suitable DEM found")

    assert out_group is not None
    fid = out_group

    # TODO: rework the tiling regime for larger dsm
    # for non single row based tiles, we won't have ideal
    # matching reads for tiled processing between the acquisition
    # and the DEM
    kwargs = compression.settings(
        filter_opts,
        chunks=(
            (1, dem_cols) if acquisition.tile_size[0] == 1 else acquisition.tile_size
        ),
    )

    group = fid.create_group(GroupName.ELEVATION_GROUP.value)

    param_grp = group.create_group("PARAMETERS")
    param_grp.attrs["left_buffer"] = margins.left
    param_grp.attrs["right_buffer"] = margins.right
    param_grp.attrs["top_buffer"] = margins.top
    param_grp.attrs["bottom_buffer"] = margins.bottom

    # dataset attributes
    attrs = {
        "crs_wkt": geobox.crs.ExportToWkt(),
        "geotransform": dem_geobox.transform.to_gdal(),
    }

    # Smooth the DSM
    dsm_data = filter_dsm(dsm_data)
    dname = DatasetName.DSM_SMOOTHED.value
    out_sm_dset = group.create_dataset(dname, data=dsm_data, **kwargs)
    desc = "A subset of a Digital Surface Model smoothed with a gaussian kernel."
    attrs["description"] = desc
    attrs["id"] = np.array([metadata["id"]], VLEN_STRING)
    attach_image_attributes(out_sm_dset, attrs)
