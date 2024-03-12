#!/usr/bin/env python
"""Digital Surface Model Data extraction and smoothing."""

import h5py
import numpy as np
import rasterio
from rasterio.warp import Resampling
from scipy import ndimage

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


def get_dsm(
    acquisition,
    srtm_pathname,
    cop_pathname,
    buffer_distance=8000,
    out_group=None,
    compression=H5CompressionFilter.LZF,
    filter_opts=None,
):
    """Given an acquisition and a national Digitial Surface Model,
    extract a subset from the DSM based on the acquisition extents
    plus an x & y margins. The subset is then smoothed with a 3x3
    gaussian filter.
    A square margins is applied to the extents.

    :param acquisition:
        An instance of an acquisition object.

    :param srtm_pathname:
        A string pathname of the SRTM DSM with a ':' to seperate the
        filename from the import HDF5 dataset name.

    :param cop_pathname:
        A string pathname of the mosaiced Copernicus 30m DEM .tif file.

    :param buffer_distance:
        A number representing the desired distance (in the same
        units as the acquisition) in which to calculate the extra
        number of pixels required to buffer an image.
        Default is 8000.

    :param out_group:
        A writeable HDF5 `Group` object.

        The dataset name will be as follows:

        * DatasetName.DSM_SMOOTHED

    :param compression:
        The compression filter to use.
        Default is H5CompressionFilter.LZF

    :filter_opts:
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
    # Use the 1st acquisition to setup the geobox
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
            dsm_geobox = GriddedGeoBox.from_dataset(dsm_ds)

            # calculate full border extents into CRS of DSM
            extents = dem_geobox.project_extents(dsm_geobox.crs)
            ul_xy = (extents[0], extents[3])
            ur_xy = (extents[2], extents[3])
            lr_xy = (extents[2], extents[1])
            ll_xy = (extents[0], extents[1])

            # load the subset and corresponding geobox
            subs, subs_geobox = read_subset(
                dsm_ds, ul_xy, ur_xy, lr_xy, ll_xy, edge_buffer=1
            )

            # ancillary metadata tracking
            metadata = current_h5_metadata(dsm_fid, dataset_path=dname)

    except (IndexError, ValueError):
        with rasterio.open(cop_pathname, "r") as dsm_ds:
            dsm_geobox = GriddedGeoBox.from_dataset(dsm_ds)

            # calculate full border extents into CRS of DSM
            extents = dem_geobox.project_extents(dsm_geobox.crs)
            ul_xy = (extents[0], extents[3])
            ur_xy = (extents[2], extents[3])
            lr_xy = (extents[2], extents[1])
            ll_xy = (extents[0], extents[1])

            # load the subset and corresponding geobox
            subs, subs_geobox = read_subset(
                dsm_ds, ul_xy, ur_xy, lr_xy, ll_xy, edge_buffer=1
            )

            # ancillary metadata tracking
            metadata = {"id": "cop-30m-dem"}

    # Retrive the DSM data
    dsm_data = reproject_array_to_array(
        subs, subs_geobox, dem_geobox, resampling=Resampling.bilinear
    )

    # free memory
    subs = None

    # Output the reprojected result
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
    desc = "A subset of a Digital Surface Model smoothed with a gaussian " "kernel."
    attrs["description"] = desc
    attrs["id"] = np.array([metadata["id"]], VLEN_STRING)
    attach_image_attributes(out_sm_dset, attrs)
