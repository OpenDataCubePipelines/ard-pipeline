#!/usr/bin/env python

"""Longitude and Latitude 2D grid creation."""

import numpy as np
from osgeo import osr

from wagl.constants import DatasetName, GroupName
from wagl.hdf5 import H5CompressionFilter, attach_image_attributes
from wagl.interpolation import interpolate_grid

CRS = "EPSG:4326"
LON_DESC = "Contains the longitude values for each pixel."
LAT_DESC = "Contains the latitude values for each pixel."


def coord_getters(geobox, geo_crs=None, centre=True):
    """
    Returns a pair of functions that when
    given an image/array y & x will return the corresponding
    longitude and latitude co-ordinates. The y, x style mimics Python indices.

    :param geobox:
        An instance of a GriddedGeoBox object.

    :param geo_crs:
        An instance of a defined geographic osr.SpatialReference
        object. If set to None (Default), then geo_crs will be set
        to WGS84.

    :param centre:
        A boolean indicating whether or not the returned co-ordinate
        should reference the centre of a pixel, in which case a 0.5
        offset is applied in the x & y directions. Default is True.

    :return:
        A function that returns the longitude and latitudes
        when given y & x integer indices.
    """
    if geo_crs is None:
        geo_crs = osr.SpatialReference()
        geo_crs.SetFromUserInput(CRS)

    affine_transform = geobox.transform

    geobox.crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    geo_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    coord_transform = osr.CoordinateTransformation(geobox.crs, geo_crs)

    def lon_func(y, x):
        if centre:
            xy = tuple(v + 0.5 for v in (x, y))
        else:
            xy = (x, y)
        mapx, mapy = affine_transform * xy
        (lon, lat, _) = coord_transform.TransformPoint(mapx, mapy)
        return lon

    def lat_func(y, x):
        if centre:
            xy = tuple(v + 0.5 for v in (x, y))
        else:
            xy = (x, y)
        mapx, mapy = affine_transform * xy
        (lon, lat, _) = coord_transform.TransformPoint(mapx, mapy)
        return lat

    return lon_func, lat_func


def create_lon_lat_grids(
    acquisition,
    out_group=None,
    compression=H5CompressionFilter.LZF,
    filter_opts=None,
    depth=7,
):
    """Creates 2 by 2D NumPy arrays containing longitude and latitude
    co-ordinates for each array element.

    :param acquisition:
        An instance of an `Acquisition` object.

    :param out_group:
        A writeable HDF5 `Group` object.

        The dataset names will be given by:

        * contants.DatasetName.LON.value
        * contants.DatasetName.LAT.value

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
    geobox = acquisition.gridded_geo_box()

    # Define the lon and lat transform funtions
    lon_func, lat_func = coord_getters(geobox=geobox)

    # Get some basic info about the image
    shape = geobox.get_shape_yx()

    # Initialise the array to contain the result
    result = np.zeros(shape, dtype="float64")
    interpolate_grid(result, lon_func, depth=depth, origin=(0, 0), shape=shape)

    assert out_group is not None
    fid = out_group

    if GroupName.LON_LAT_GROUP.value not in fid:
        fid.create_group(GroupName.LON_LAT_GROUP.value)

    grp = fid[GroupName.LON_LAT_GROUP.value]

    # define some base attributes for the image datasets
    attrs = {
        "crs_wkt": geobox.crs.ExportToWkt(),
        "geotransform": geobox.transform.to_gdal(),
        "description": LON_DESC,
    }

    kwargs = compression.settings(filter_opts, chunks=acquisition.tile_size)
    lon_dset = grp.create_dataset(DatasetName.LON.value, data=result, **kwargs)
    attach_image_attributes(lon_dset, attrs)

    result = np.zeros(shape, dtype="float64")
    interpolate_grid(result, lat_func, depth=depth, origin=(0, 0), shape=shape)

    attrs["description"] = LAT_DESC
    lat_dset = grp.create_dataset(DatasetName.LAT.value, data=result, **kwargs)
    attach_image_attributes(lat_dset, attrs)
