#!/usr/bin/env python

"""Functions for calculating cast shadow from both the sun and satellite
---------------------------------------------------------------------.

as source directions, as well as self shadow masks.
---------------------------------------------------
"""

import numpy as np

from wagl.__cast_shadow_mask import cast_shadow_main as cast_shadow_prim
from wagl.constants import DatasetName, GroupName
from wagl.geobox import GriddedGeoBox
from wagl.hdf5 import H5CompressionFilter, attach_image_attributes
from wagl.margins import pixel_buffer
from wagl.tiling import generate_tiles


def self_shadow(
    incident_angles_group,
    exiting_angles_group,
    out_group=None,
    compression=H5CompressionFilter.LZF,
    filter_opts=None,
):
    """Computes the self shadow mask.

    :param incident_angles_group:
        The root HDF5 `Group` that contains the incident
        angle dataset specified by the pathname given by:

        * DatasetName.INCIDENT

    :param exiting_angles_group:
        The root HDF5 `Group` that contains the exiting
        angle dataset specified by the pathname given by:

        * DatasetName.EXITING

    :param out_group:
        A writeable HDF5 `Group` object.
        The dataset name will be given by:

        * DatasetName.SELF_SHADOW

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
    incident_angle = incident_angles_group[DatasetName.INCIDENT.value]
    exiting_angle = exiting_angles_group[DatasetName.EXITING.value]
    geobox = GriddedGeoBox.from_dataset(incident_angle)

    assert out_group is not None
    fid = out_group

    if GroupName.SHADOW_GROUP.value not in fid:
        fid.create_group(GroupName.SHADOW_GROUP.value)

    grp = fid[GroupName.SHADOW_GROUP.value]

    tile_size = exiting_angle.chunks
    kwargs = compression.settings(filter_opts, chunks=tile_size)
    cols, rows = geobox.get_shape_xy()
    kwargs["shape"] = (rows, cols)
    kwargs["dtype"] = "bool"

    # output dataset
    dataset_name = DatasetName.SELF_SHADOW.value
    out_dset = grp.create_dataset(dataset_name, **kwargs)

    # attach some attributes to the image datasets
    attrs = {
        "crs_wkt": geobox.crs.ExportToWkt(),
        "geotransform": geobox.transform.to_gdal(),
    }
    desc = "Self shadow mask derived using the incident and exiting angles."
    attrs["description"] = desc
    attrs["alias"] = "self-shadow"
    attach_image_attributes(out_dset, attrs)

    # process by tile
    for tile in generate_tiles(cols, rows, tile_size[1], tile_size[0]):
        # Row and column start locations
        ystart, yend = tile[0]
        xstart, xend = tile[1]
        idx = (slice(ystart, yend), slice(xstart, xend))

        # Read the data for the current tile
        inc = np.radians(incident_angle[idx])
        exi = np.radians(exiting_angle[idx])

        # Process the tile
        mask = np.ones(inc.shape, dtype="uint8")
        mask[np.cos(inc) <= 0.0] = 0
        mask[np.cos(exi) <= 0.0] = 0

        # Write the current tile to disk
        out_dset[idx] = mask


class FortranError(Exception):
    """Base class for errors thrown from the Fortran code used in this module."""

    def __init__(self, function_name, code, msg):
        self.function_name = function_name
        self.code = code
        self.msg = msg or "Unknown error"

    def __str__(self):
        """Return a string representation of this Error."""
        err = "Error in Fotran code {0} (code {1}): {2}"
        err = err.format(self.function_name, self.code, self.msg)
        return err


class CastShadowError(FortranError):
    """Class that deals with errors from :py:func:`calculate_cast_shadow`."""

    def __init__(self, code):
        super().__init__(
            "cast_shadow_main", code, CastShadowError.get_error_message(code)
        )

    @staticmethod
    def get_error_message(code):
        """Generate an error message for a specific code. It is OK for this have
        non-returning control paths, as this will result in ``None``, which
        is handled in the super class.
        """

        def tmpt(d, n):
            """Generate message."""
            err = f"attempt to access invalid {d} of {n}"
            return err

        if code == 20:
            return tmpt("x", "dem")
        if code == 21:
            return tmpt("x", "dem_data")
        if code == 22:
            return tmpt("x", "solar and sazi")
        if code == 23:
            return tmpt("x", "solar_data")
        if code == 24:
            return tmpt("x", "a")
        if code == 25:
            return tmpt("y", "dem_data")
        if code == 26:
            return tmpt("y", "a")
        if code == 27:
            return tmpt("x", "mask_all")
        if code == 28:
            return tmpt("y", "mask_all")
        if code == 29:
            return tmpt("x", "mask")
        if code == 30:
            return tmpt("y", "mask")
        if code == 31:
            return tmpt("X", "dem and a")
        if code == 32:
            return tmpt("y", "a")
        if code == 33:
            return tmpt("y", "dem")
        if code == 34:
            return tmpt("x", "mask_all")
        if code == 35:
            return tmpt("x", "mask")
        if code == 36:
            return tmpt("y", "mask_all")
        if code == 37:
            return tmpt("y", "mask")
        if code == 38:
            return tmpt("x", "dem")
        if code == 39:
            return tmpt("x", "dem_data")
        if code == 40:
            return tmpt("x", "solar")
        if code == 41:
            return tmpt("x", "solar_data")
        if code == 42:
            return tmpt("x", "a and dem")
        if code == 43:
            return tmpt("y", "a")
        if code == 44:
            return tmpt("y", "dem")
        if code == 45:
            return tmpt("x", "mask_all")
        if code == 46:
            return tmpt("x", "mask")
        if code == 47:
            return tmpt("y", "mask_alll")
        if code == 48:
            return tmpt("y", "mask")
        if code == 49:
            return tmpt("x", "a and dem")
        if code == 50:
            return tmpt("y", "a")
        if code == 51:
            return tmpt("y", "dem")
        if code == 52:
            return tmpt("x", "mask_all")
        if code == 53:
            return tmpt("x", "mask")
        if code == 54:
            return tmpt("y", "mask_all")
        if code == 55:
            return tmpt("y", "mask")
        if code == 61:
            return "azimuth case not possible - phi_sun must be in 0 to 360 deg"
        if code == 62:
            return "k_max gt k_setting"
        if code == 63:
            return "add outside add_max ranges"
        if code == 71:
            return "Parameters defining A are invalid"
        if code == 72:
            return "Matrix A not embedded in image"
        if code == 73:
            return "matrix A does not have sufficient y margin"
        if code == 74:
            return "matrix A does not have sufficient x margin"


def cast_shadow_main(
    dem_data,
    solar_data,
    sazi_data,
    dresx,
    dresy,
    aoff_x1,
    aoff_x2,
    aoff_y1,
    aoff_y2,
    nla_ori,
    nsa_ori,
    mask_all,
):
    # some names are from the Fortran code and so does not follow Python conventions
    nrow, ncol = solar_data.shape
    nl, ns = dem_data.shape

    sources = dem_data.attrs["id"]
    if sources.shape == (1,) and sources[0] == "cop-30m-dem":
        htol = 1.0
        sun_disk = 0.0
    else:
        htol = 1.0
        sun_disk = 3.0

    # the DEM data should have the same dimensions as the angle data
    # but with padding
    assert nl == aoff_y1 + nrow + aoff_y2
    assert ns == aoff_x1 + ncol + aoff_x2

    y_indices = [
        slice(row, min(row + nla_ori, nrow)) for row in range(0, nrow, nla_ori)
    ]
    for y_idx in y_indices:
        nla = y_idx.stop - y_idx.start
        mmax_sub = aoff_y1 + nla + aoff_y2

        solar = solar_data[y_idx, :]
        sazi = sazi_data[y_idx, :]
        dem = dem_data[y_idx.start : (y_idx.start + mmax_sub), :]

        ierr, _, mask = cast_shadow_prim(
            dem,
            solar,
            sazi,
            dresx,
            dresy,
            aoff_x1,
            aoff_x2,
            aoff_y1,
            aoff_y2,
            nla_ori,
            nsa_ori,
            htol,
            sun_disk,
        )

        if ierr:
            raise CastShadowError(ierr)

        mask_all[y_idx, :] = mask


def calculate_cast_shadow(
    acquisition,
    dsm_group,
    satellite_solar_group,
    buffer_distance,
    out_group=None,
    compression=H5CompressionFilter.LZF,
    filter_opts=None,
    solar_source=True,
):
    """
    This code is an interface to the fortran code
    cast_shadow_main.f90 written by Fuqin Li (and modified to
    work with F2py).

    The following was taken from the top of the Fotran program:
    "cast_shadow_main.f90":

    Creates a shadow mask for a standard Landsat scene
    the program was originally written by DLB Jupp in Oct. 2010
    for a small sub_matrix and was modified by Fuqin Li in Oct.
    2010 so that the program can be used for large landsat scene.

    Basically, a sub-matrix A is embedded in a larger DEM image
    and the borders must be large enough to find the shaded pixels.
    If we assume the solar azimuth and zenith angles change very
    little within the sub-matrix A, then the Landsat scene can be
    divided into several sub_matrix.
    For Australian region, with 0 .00025 degree resolution, the
    sub-matrix A is set to 500x500

    we also need to set extra DEM lines/columns to run the Landsat
    scene. This will change with elevation
    difference within the scene and solar zenith angle. For
    Australian region and Landsat scene with 0.00025 degree
    resolution, the maximum extra lines are set to 250 pixels/lines
    for each direction. This figure should be sufficient for everywhere
    and anytime in Australia. Thus, the DEM image will be larger than
    landsat image for 500 lines x 500 columns

    :param acquisition:
        An instance of an acquisition object.

    :param dsm_group:
        The root HDF5 `Group` that contains the Digital Surface Model
        data.
        The dataset pathnames are given by:

        * DatasetName.DSM_SMOOTHED

        The dataset must have the same dimensions as `acquisition`
        plus a margin of widths specified by margin.

    :param satellite_solar_group:
        The root HDF5 `Group` that contains the satellite and solar
        datasets specified by the pathnames given by:

        * DatasetName.SOLAR_ZENITH
        * DatasetName.SOLAR_AZIMUTH
        * DatasetName.SATELLITE_VIEW
        * DatasetName.SATELLITE_AZIMUTH

    :param buffer_distance:
        A number representing the desired distance (in the same
        units as the acquisition) in which to calculate the extra
        number of pixels required to buffer an image.

    :param out_group:
        A writeable HDF5 `Group` object.

        The dataset names will be given by the format string detailed
        by:

        * DatasetName.CAST_SHADOW_FMT

    :param compression:
        The compression filter to use. Default is H5CompressionFilter.LZF.

    :param filter_opts:
        A dict of key value pairs available to the given configuration
        instance of H5CompressionFilter. For example
        H5CompressionFilter.LZF has the keywords *chunks* and *shuffle*
        available.
        Default is None, which will use the default settings for the
        chosen H5CompressionFilter instance.

    :param solar_source:
        A `bool` indicating if the source for the line of sight comes
        from the sun (True; Default), or False indicating the satellite.

    :return:
        An opened `h5py.File` object, that is either in-memory using the
        `core` driver, or on disk.

    :warning:
        The Fortran code cannot be compiled with ``-O3`` as it
        produces incorrect results if it is.
    """
    # Setup the geobox
    geobox = acquisition.gridded_geo_box()
    x_res, y_res = geobox.pixelsize

    # Define Top, Bottom, Left, Right pixel buffer margins
    margins = pixel_buffer(acquisition, buffer_distance)

    if solar_source:
        zenith_name = DatasetName.SOLAR_ZENITH.value
        azimuth_name = DatasetName.SOLAR_AZIMUTH.value
    else:
        zenith_name = DatasetName.SATELLITE_VIEW.value
        azimuth_name = DatasetName.SATELLITE_AZIMUTH.value

    zenith_angle = satellite_solar_group[zenith_name]
    azimuth_angle = satellite_solar_group[azimuth_name]
    elevation = dsm_group[DatasetName.DSM_SMOOTHED.value]

    # block height and width of the window/sub-matrix used in the cast
    # shadow algorithm
    block_width = margins.left + margins.right
    block_height = margins.top + margins.bottom

    source_dir = "SUN" if solar_source else "SATELLITE"

    assert out_group is not None
    fid = out_group

    if GroupName.SHADOW_GROUP.value not in fid:
        fid.create_group(GroupName.SHADOW_GROUP.value)

    grp = fid[GroupName.SHADOW_GROUP.value]
    tile_size = satellite_solar_group[zenith_name].chunks
    kwargs = compression.settings(filter_opts, chunks=tile_size)
    kwargs["dtype"] = "bool"

    dname_fmt = DatasetName.CAST_SHADOW_FMT.value
    out_dset = grp.create_dataset(
        dname_fmt.format(source=source_dir),
        shape=zenith_angle.shape,
        **kwargs,
    )

    # attach some attributes to the image datasets
    attrs = {
        "crs_wkt": geobox.crs.ExportToWkt(),
        "geotransform": geobox.transform.to_gdal(),
    }
    desc = (
        f"The cast shadow mask determined using the {source_dir} "
        "as the source direction."
    )
    attrs["description"] = desc
    attrs["alias"] = f"cast-shadow-{source_dir}".lower()
    attach_image_attributes(out_dset, attrs)

    # Compute the cast shadow mask
    cast_shadow_main(
        elevation,
        zenith_angle,
        azimuth_angle,
        x_res,
        y_res,
        margins.left,
        margins.right,
        margins.top,
        margins.bottom,
        block_height,
        block_width,
        out_dset,
    )


def combine_shadow_masks(
    self_shadow_group,
    cast_shadow_sun_group,
    cast_shadow_satellite_group,
    out_group=None,
    compression=H5CompressionFilter.LZF,
    filter_opts=None,
):
    """A convenience function for combining the shadow masks into a single
    boolean array.

    :param self_shadow_group:
        The root HDF5 `Group` that contains the self shadow
        dataset specified by the pathname given by:

        * DatasetName.SELF_SHADOW

    :param cast_shadow_sun_group:
        The root HDF5 `Group` that contains the cast shadow
        (solar direction) dataset specified by the pathname
        given by:

        * DatasetName.CAST_SHADOW_FMT

    :param cast_shadow_satellite_group:
        The root HDF5 `Group` that contains the cast shadow
        (satellite direction) dataset specified by the pathname
        given by:

        * DatasetName.CAST_SHADOW_FMT

    :param out_group:
        A writeable HDF5 `Group` object.

        The dataset names will be given by the format string detailed
        by:

        * DatasetName.COMBINED_SHADOW

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
    # access the datasets
    dname_fmt = DatasetName.CAST_SHADOW_FMT.value
    self_shad = self_shadow_group[DatasetName.SELF_SHADOW.value]
    cast_sun = cast_shadow_sun_group[dname_fmt.format(source="SUN")]
    dname = dname_fmt.format(source="SATELLITE")
    cast_sat = cast_shadow_satellite_group[dname]
    geobox = GriddedGeoBox.from_dataset(self_shad)

    assert out_group is not None
    fid = out_group

    if GroupName.SHADOW_GROUP.value not in fid:
        fid.create_group(GroupName.SHADOW_GROUP.value)

    grp = fid[GroupName.SHADOW_GROUP.value]
    tile_size = cast_sun.chunks
    kwargs = compression.settings(filter_opts, chunks=tile_size)
    cols, rows = geobox.get_shape_xy()
    kwargs["shape"] = (rows, cols)
    kwargs["dtype"] = "bool"

    # output dataset
    out_dset = grp.create_dataset(DatasetName.COMBINED_SHADOW.value, **kwargs)

    # attach some attributes to the image datasets
    attrs = {
        "crs_wkt": geobox.crs.ExportToWkt(),
        "geotransform": geobox.transform.to_gdal(),
    }
    desc = (
        "Combined shadow masks: 1. self shadow, "
        "2. cast shadow (solar direction), "
        "3. cast shadow (satellite direction)."
    )
    attrs["description"] = desc
    attrs["mask_values"] = "False = Shadow; True = Non Shadow"
    attrs["alias"] = "terrain-shadow"
    attach_image_attributes(out_dset, attrs)

    # process by tile
    for tile in generate_tiles(cols, rows, tile_size[1], tile_size[0]):
        # Row and column start locations
        ystart, yend = tile[0]
        xstart, xend = tile[1]
        idx = (slice(ystart, yend), slice(xstart, xend))

        out_dset[idx] = self_shad[idx] & cast_sun[idx] & cast_sat[idx]
