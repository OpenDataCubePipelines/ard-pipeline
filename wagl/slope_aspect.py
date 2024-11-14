#!/usr/bin/env python

"""Calculates the slope and aspect for a given elevation dataset."""

import numpy as np

from wagl.__slope_aspect import slope_aspect as slope_aspect_prim
from wagl.constants import DatasetName, GroupName
from wagl.data import as_array
from wagl.hdf5 import H5CompressionFilter, attach_image_attributes
from wagl.margins import pixel_buffer
from wagl.satellite_solar_angles import setup_spheroid


def slope_aspect(
    nrow,
    ncol,
    nrow_alloc,
    ncol_alloc,
    dresx,
    dresy,
    spheroid,
    alat,
    is_utm,
    dem,
    theta,
    phit,
):
    slope_aspect_prim(
        nrow,
        ncol,
        nrow_alloc,
        ncol_alloc,
        dresx,
        dresy,
        spheroid,
        alat,
        is_utm,
        dem,
        theta,
        phit,
    )


def slope_aspect_arrays(
    acquisition,
    dsm_group,
    buffer_distance,
    out_group=None,
    compression=H5CompressionFilter.LZF,
    filter_opts=None,
):
    """Calculates slope and aspect.

    :param acquisition:
        An instance of an acquisition object.

    :param dsm_group:
        The root HDF5 `Group` that contains the Digital Surface Model
        data.
        The dataset pathname is given by:

        * DatasetName.DSM_SMOOTHED

        The dataset must have the same dimensions as `acquisition`
        plus a margin of widths specified by margin.

    :param buffer_distance:
        A number representing the desired distance (in the same
        units as the acquisition) in which to calculate the extra
        number of pixels required to buffer an image.

    :param out_group:
        A writeable HDF5 `Group` object.

        The dataset names will be given by the format string detailed
        by:

        * DatasetName.SLOPE
        * DatasetName.ASPECT

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
    # Setup the geobox
    geobox = acquisition.gridded_geo_box()

    # Retrive the spheroid parameters
    # (used in calculating pixel size in metres per lat/lon)
    spheroid, _ = setup_spheroid(geobox.crs.ExportToWkt())

    # Are we in projected or geographic space
    is_utm = not geobox.crs.IsGeographic()

    # Define Top, Bottom, Left, Right pixel margins
    margins = pixel_buffer(acquisition, buffer_distance)

    # Get the x and y pixel sizes
    _, y_origin = geobox.origin
    x_res, y_res = geobox.pixelsize

    # Get acquisition dimensions and add 1 pixel top, bottom, left & right
    cols, rows = geobox.get_shape_xy()
    ncol = cols + 2
    nrow = rows + 2

    # elevation dataset
    elevation = dsm_group[DatasetName.DSM_SMOOTHED.value]
    ele_rows, ele_cols = elevation.shape

    # TODO: check that the index is correct
    # Define the index to read the DEM subset
    ystart, ystop = (margins.top - 1, ele_rows - (margins.bottom - 1))
    xstart, xstop = (margins.left - 1, ele_cols - (margins.right - 1))
    idx = (slice(ystart, ystop), slice(xstart, xstop))

    subset = as_array(elevation[idx], dtype=np.float32, transpose=True)

    # Define an array of latitudes
    # This will be ignored if is_utm == True
    alat = np.array(
        [y_origin - i * y_res for i in range(-1, nrow - 1)], dtype=np.float64
    )  # yes, I did mean float64.

    # Output the reprojected result
    assert out_group is not None
    fid = out_group

    if GroupName.SLP_ASP_GROUP.value not in fid:
        fid.create_group(GroupName.SLP_ASP_GROUP.value)

    group = fid[GroupName.SLP_ASP_GROUP.value]

    # metadata for calculation
    param_group = group.create_group("PARAMETERS")
    param_group.attrs["dsm_index"] = ((ystart, ystop), (xstart, xstop))
    param_group.attrs["pixel_buffer"] = "1 pixel"

    kwargs = compression.settings(filter_opts, chunks=acquisition.tile_size)
    no_data = -999
    kwargs["fillvalue"] = no_data

    # Define the output arrays. These will be transposed upon input
    slope = np.zeros((rows, cols), dtype="float32")
    aspect = np.zeros((rows, cols), dtype="float32")

    slope_aspect(
        ncol,
        nrow,
        cols,
        rows,
        x_res,
        y_res,
        spheroid,
        alat,
        is_utm,
        subset,
        slope.transpose(),
        aspect.transpose(),
    )

    # output datasets
    dname = DatasetName.SLOPE.value
    slope_dset = group.create_dataset(dname, data=slope, **kwargs)
    dname = DatasetName.ASPECT.value
    aspect_dset = group.create_dataset(dname, data=aspect, **kwargs)

    # attach some attributes to the image datasets
    attrs = {
        "crs_wkt": geobox.crs.ExportToWkt(),
        "geotransform": geobox.transform.to_gdal(),
        "no_data_value": no_data,
    }
    desc = "The slope derived from the input elevation model."
    attrs["description"] = desc
    attach_image_attributes(slope_dset, attrs)

    desc = "The aspect derived from the input elevation model."
    attrs["description"] = desc
    attach_image_attributes(aspect_dset, attrs)
