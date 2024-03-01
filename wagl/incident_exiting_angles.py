#!/usr/bin/env python

"""Calculates 2D grids of incident, exiting and relative azimuthal angles."""

import numpy as np

from wagl.constants import DatasetName, GroupName
from wagl.data import as_array
from wagl.geobox import GriddedGeoBox
from wagl.hdf5 import H5CompressionFilter, attach_image_attributes
from wagl.tiling import generate_tiles


def angles_off_of_sloped_terrain(
    theta,
    phi,
    slope,
    aspect,
):
    cos_slope = np.cos(slope)
    sin_slope = np.sin(slope)

    cos_theta = np.cos(theta)
    sin_theta = np.cos(theta)

    pdiff = phi - aspect
    cos_pdiff = np.cos(pdiff)
    sin_pdiff = np.sin(pdiff)

    cos_theta_out = cos_theta * cos_slope + sin_theta * sin_slope * cos_pdiff
    cos_theta_out[cos_theta_out >= 1.0] = 1.0
    theta_out = np.arccos(cos_theta_out)

    # this is not quite phi_out yet
    # there's going to be an offset added to this
    sin_phi_o = sin_theta * sin_pdiff
    cos_phi_o = cos_theta * sin_slope - sin_theta * cos_slope * cos_pdiff

    offset = np.arctan(np.tan(np.pi - aspect) * cos_slope)
    phi_out = np.arctan2(sin_phi_o, cos_phi_o) - offset

    # standard interval of (-pi, pi]
    theta_out[theta_out <= -np.pi] += 2 * np.pi
    theta_out[theta_out > np.pi] -= 2 * np.pi
    phi_out[phi_out <= -np.pi] += 2 * np.pi
    phi_out[phi_out > np.pi] -= 2 * np.pi

    return theta_out, phi_out


def incident_angle(
    solar,
    sazi,
    theta,
    phit,
):
    return angles_off_of_sloped_terrain(
        solar,
        sazi,
        theta,
        phit,
    )


def exiting_angle(
    view,
    azi,
    theta,
    phit,
):
    return angles_off_of_sloped_terrain(
        view,
        azi,
        theta,
        phit,
    )


def incident_angles(
    satellite_solar_group,
    slope_aspect_group,
    out_group=None,
    compression=H5CompressionFilter.LZF,
    filter_opts=None,
):
    """Calculates the incident angle and the azimuthal incident angle.

    :param satellite_solar_group:
        The root HDF5 `Group` that contains the solar zenith and
        solar azimuth datasets specified by the pathnames given by:

        * DatasetName.SOLAR_ZENITH
        * DatasetName.SOLAR_AZIMUTH

    :param slope_aspect_group:
        The root HDF5 `Group` that contains the slope and aspect
        datasets specified by the pathnames given by:

        * DatasetName.SLOPE
        * DatasetName.ASPECT

    :param out_group:
        A writeable HDF5 `Group` object.

        The dataset names will be as follows:

        * DatasetName.INCIDENT
        * DatasetName.AZIMUTHAL_INCIDENT

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
    # dataset arrays
    dname = DatasetName.SOLAR_ZENITH.value
    solar_zenith_dataset = satellite_solar_group[dname]
    dname = DatasetName.SOLAR_AZIMUTH.value
    solar_azimuth_dataset = satellite_solar_group[dname]
    slope_dataset = slope_aspect_group[DatasetName.SLOPE.value]
    aspect_dataset = slope_aspect_group[DatasetName.ASPECT.value]

    geobox = GriddedGeoBox.from_dataset(solar_zenith_dataset)
    shape = geobox.get_shape_yx()
    rows, cols = shape
    crs = geobox.crs.ExportToWkt()

    assert out_group is not None
    fid = out_group

    if GroupName.INCIDENT_GROUP.value not in fid:
        fid.create_group(GroupName.INCIDENT_GROUP.value)

    if filter_opts is None:
        filter_opts = {}

    grp = fid[GroupName.INCIDENT_GROUP.value]
    tile_size = solar_zenith_dataset.chunks
    filter_opts["chunks"] = tile_size
    kwargs = compression.config(**filter_opts).dataset_compression_kwargs()
    no_data = np.nan
    kwargs["shape"] = shape
    kwargs["fillvalue"] = no_data
    kwargs["dtype"] = "float32"

    # output datasets
    dataset_name = DatasetName.INCIDENT.value
    incident_dset = grp.create_dataset(dataset_name, **kwargs)
    dataset_name = DatasetName.AZIMUTHAL_INCIDENT.value
    azi_inc_dset = grp.create_dataset(dataset_name, **kwargs)

    # attach some attributes to the image datasets
    attrs = {
        "crs_wkt": crs,
        "geotransform": geobox.transform.to_gdal(),
        "no_data_value": no_data,
    }
    desc = "Contains the incident angles in degrees."
    attrs["description"] = desc
    attrs["alias"] = "incident"
    attach_image_attributes(incident_dset, attrs)

    desc = "Contains the azimuthal incident angles in degrees."
    attrs["description"] = desc
    attrs["alias"] = "azimuthal-incident"
    attach_image_attributes(azi_inc_dset, attrs)

    # process by tile
    for tile in generate_tiles(cols, rows, tile_size[1], tile_size[0]):
        # Row and column start and end locations
        ystart = tile[0][0]
        xstart = tile[1][0]
        yend = tile[0][1]
        xend = tile[1][1]
        idx = (slice(ystart, yend), slice(xstart, xend))

        # Read the data for the current tile
        # Convert to required datatype and transpose
        sol_zen = as_array(solar_zenith_dataset[idx], dtype=np.float32, transpose=True)
        sol_azi = as_array(solar_azimuth_dataset[idx], dtype=np.float32, transpose=True)
        slope = as_array(slope_dataset[idx], dtype=np.float32, transpose=True)
        aspect = as_array(aspect_dataset[idx], dtype=np.float32, transpose=True)

        # Process the current tile
        incident, azi_incident = incident_angle(
            np.radians(sol_zen),
            np.radians(sol_azi),
            np.radians(slope),
            np.radians(aspect),
        )

        # Write the current tile to disk
        incident_dset[idx] = np.degrees(incident.transpose())
        azi_inc_dset[idx] = np.degrees(azi_incident.transpose())


def exiting_angles(
    satellite_solar_group,
    slope_aspect_group,
    out_group=None,
    compression=H5CompressionFilter.LZF,
    filter_opts=None,
):
    """Calculates the exiting angle and the azimuthal exiting angle.

    :param satellite_solar_group:
        The root HDF5 `Group` that contains the satellite view and
        satellite azimuth datasets specified by the pathnames given by:

        * DatasetName.SATELLITE_VIEW
        * DatasetName.SATELLITE_AZIMUTH

    :param slope_aspect_group:
        The root HDF5 `Group` that contains the slope and aspect
        datasets specified by the pathnames given by:

        * DatasetName.SLOPE
        * DatasetName.ASPECT

    :param out_group:
        A writeable HDF5 `Group` object.

        The dataset names will be as follows:

        * DatasetName.EXITING
        * DatasetName.AZIMUTHAL_EXITING

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
    # dataset arrays
    dname = DatasetName.SATELLITE_VIEW.value
    satellite_view_dataset = satellite_solar_group[dname]
    dname = DatasetName.SATELLITE_AZIMUTH.value
    satellite_azimuth_dataset = satellite_solar_group[dname]
    slope_dataset = slope_aspect_group[DatasetName.SLOPE.value]
    aspect_dataset = slope_aspect_group[DatasetName.ASPECT.value]

    geobox = GriddedGeoBox.from_dataset(satellite_view_dataset)
    shape = geobox.get_shape_yx()
    rows, cols = shape
    crs = geobox.crs.ExportToWkt()

    assert out_group is not None
    fid = out_group

    if GroupName.EXITING_GROUP.value not in fid:
        fid.create_group(GroupName.EXITING_GROUP.value)

    if filter_opts is None:
        filter_opts = {}

    grp = fid[GroupName.EXITING_GROUP.value]
    tile_size = satellite_view_dataset.chunks
    filter_opts["chunks"] = tile_size
    kwargs = compression.config(**filter_opts).dataset_compression_kwargs()
    no_data = np.nan
    kwargs["shape"] = shape
    kwargs["fillvalue"] = no_data
    kwargs["dtype"] = "float32"

    # output datasets
    dataset_name = DatasetName.EXITING.value
    exiting_dset = grp.create_dataset(dataset_name, **kwargs)
    dataset_name = DatasetName.AZIMUTHAL_EXITING.value
    azi_exit_dset = grp.create_dataset(dataset_name, **kwargs)

    # attach some attributes to the image datasets
    attrs = {
        "crs_wkt": crs,
        "geotransform": geobox.transform.to_gdal(),
        "no_data_value": no_data,
    }
    desc = "Contains the exiting angles in degrees."
    attrs["description"] = desc
    attrs["alias"] = "exiting"
    attach_image_attributes(exiting_dset, attrs)

    desc = "Contains the azimuthal exiting angles in degrees."
    attrs["description"] = desc
    attrs["alias"] = "azimuthal-exiting"
    attach_image_attributes(azi_exit_dset, attrs)

    # process by tile
    for tile in generate_tiles(cols, rows, tile_size[1], tile_size[0]):
        # Row and column start and end locations
        ystart = tile[0][0]
        xstart = tile[1][0]
        yend = tile[0][1]
        xend = tile[1][1]
        idx = (slice(ystart, yend), slice(xstart, xend))

        # Read the data for the current tile
        # Convert to required datatype and transpose
        sat_view = as_array(
            satellite_view_dataset[idx], dtype=np.float32, transpose=True
        )
        sat_azi = as_array(
            satellite_azimuth_dataset[idx], dtype=np.float32, transpose=True
        )
        slope = as_array(slope_dataset[idx], dtype=np.float32, transpose=True)
        aspect = as_array(aspect_dataset[idx], dtype=np.float32, transpose=True)

        # Process the current tile
        exiting, azi_exiting = exiting_angle(
            np.radians(sat_view),
            np.radians(sat_azi),
            np.radians(slope),
            np.radians(aspect),
        )

        # Write the current to disk
        exiting_dset[idx] = np.degrees(exiting.transpose())
        azi_exit_dset[idx] = np.degrees(azi_exiting.transpose())


def relative_azimuth_slope(
    incident_angles_group,
    exiting_angles_group,
    out_group=None,
    compression=H5CompressionFilter.LZF,
    filter_opts=None,
):
    """Calculates the relative azimuth angle on the slope surface.

    :param incident_angles_group:
        The root HDF5 `Group` that contains the azimuthal incident
        angle dataset specified by the pathname given by:

        * DatasetName.AZIMUTHAL_INCIDENT

    :param exiting_angles_group:
        The root HDF5 `Group` that contains the azimuthal exiting
        angle dataset specified by the pathname given by:

        * DatasetName.AZIMUTHAL_EXITING

    :param out_group:
        A writeable HDF5 `Group` object.

        The dataset names will be as follows:

        * DatasetName.RELATIVE_SLOPE

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
    # dataset arrays
    dname = DatasetName.AZIMUTHAL_INCIDENT.value
    azimuth_incident_dataset = incident_angles_group[dname]
    dname = DatasetName.AZIMUTHAL_EXITING.value
    azimuth_exiting_dataset = exiting_angles_group[dname]

    geobox = GriddedGeoBox.from_dataset(azimuth_incident_dataset)
    shape = geobox.get_shape_yx()
    rows, cols = shape
    crs = geobox.crs.ExportToWkt()

    assert out_group is not None
    fid = out_group

    if GroupName.REL_SLP_GROUP.value not in fid:
        fid.create_group(GroupName.REL_SLP_GROUP.value)

    if filter_opts is None:
        filter_opts = {}

    grp = fid[GroupName.REL_SLP_GROUP.value]
    tile_size = azimuth_incident_dataset.chunks
    filter_opts["chunks"] = tile_size
    kwargs = compression.config(**filter_opts).dataset_compression_kwargs()
    no_data = np.nan
    kwargs["shape"] = shape
    kwargs["fillvalue"] = no_data
    kwargs["dtype"] = "float32"

    # output datasets
    out_dset = grp.create_dataset(DatasetName.RELATIVE_SLOPE.value, **kwargs)

    # attach some attributes to the image datasets
    attrs = {
        "crs_wkt": crs,
        "geotransform": geobox.transform.to_gdal(),
        "no_data_value": no_data,
    }
    desc = "Contains the relative azimuth angles on the slope surface in " "degrees."
    attrs["description"] = desc
    attrs["alias"] = "relative-slope"
    attach_image_attributes(out_dset, attrs)

    # process by tile
    for tile in generate_tiles(cols, rows, tile_size[1], tile_size[0]):
        # Row and column start and end locations
        ystart, yend = tile[0]
        xstart, xend = tile[1]
        idx = (slice(ystart, yend), slice(xstart, xend))

        # Read the data for the current tile
        azi_inc = azimuth_incident_dataset[idx]
        azi_exi = azimuth_exiting_dataset[idx]

        # Process the tile
        rel_azi = azi_inc - azi_exi
        rel_azi[rel_azi <= -180.0] += 360.0
        rel_azi[rel_azi > 180.0] -= 360.0

        # Write the current tile to disk
        out_dset[idx] = rel_azi
