"""Calculates the Lambertian, BRDF corrected and BRDF + Terrain corrected
----------------------------------------------------------------------.

reflectance
-----------
"""

import numpy as np

from wagl.__surface_reflectance import reflectance as reflectance_prim
from wagl.constants import ArdProducts as AP
from wagl.constants import AtmosphericCoefficients as AC
from wagl.constants import BrdfDirectionalParameters, DatasetName, GroupName
from wagl.data import as_array
from wagl.hdf5 import (
    H5CompressionFilter,
    attach_image_attributes,
)


def reflectance_python(
    nrow,
    ncol,
    rori,
    norm_1,
    norm_2,
    ref_adj,
    no_data,
    radiance,
    shadow_mask,
    solar_angle,
    sazi_angle,
    view_angle,
    rela_angle,
    slope_angle,
    aspect_angle,
    it_angle,
    et_angle,
    rela_slope,
    a_mod,
    b_mod,
    s_mod,
    fs,
    fv,
    ts,
    edir_h,
    edif_h,
    ref_lm,
    ref_brdf,
    ref_terrain,
    iref_lm,
    iref_brdf,
    iref_terrain,
    norm_solar_zenith,
):
    reflectance_prim(
        nrow,
        ncol,
        rori,
        norm_1,
        norm_2,
        ref_adj,
        no_data,
        radiance,
        shadow_mask,
        solar_angle,
        sazi_angle,
        view_angle,
        rela_angle,
        slope_angle,
        aspect_angle,
        it_angle,
        et_angle,
        rela_slope,
        a_mod,
        b_mod,
        s_mod,
        fs,
        fv,
        ts,
        edir_h,
        edif_h,
        ref_lm,
        ref_brdf,
        ref_terrain,
        iref_lm,
        iref_brdf,
        iref_terrain,
        norm_solar_zenith,
    )


NO_DATA_VALUE = -999


def calculate_reflectance(
    acquisition,
    interpolation_group,
    satellite_solar_group,
    slope_aspect_group,
    relative_slope_group,
    incident_angles_group,
    exiting_angles_group,
    shadow_masks_group,
    ancillary_group,
    rori,
    out_group=None,
    compression=H5CompressionFilter.LZF,
    filter_opts=None,
    normalized_solar_zenith=45.0,
    esun=None,
):
    """Calculates Lambertian, BRDF corrected and BRDF + terrain
    illumination corrected surface reflectance.

    :param acquisition:
        An instance of an acquisition object.

    :param interpolation_group:
        The root HDF5 `Group` that contains the interpolated
        atmospheric coefficients.
        The dataset pathnames are given by:

        * DatasetName.INTERPOLATION_FMT

    :param satellite_solar_group:
        The root HDF5 `Group` that contains the solar zenith and
        solar azimuth datasets specified by the pathnames given by:

        * DatasetName.SOLAR_ZENITH
        * DatasetName.SOLAR_AZIMUTH
        * DatasetName.SATELLITE_VIEW
        * DatasetName.SATELLITE_AZIMUTH
        * DatasetName.RELATIVE_AZIMUTH

    :param slope_aspect_group:
        The root HDF5 `Group` that contains the slope and aspect
        datasets specified by the pathnames given by:

        * DatasetName.SLOPE
        * DatasetName.ASPECT

    :param relative_slope_group:
        The root HDF5 `Group` that contains the relative slope dataset
        specified by the pathname given by:

        * DatasetName.RELATIVE_SLOPE

    :param incident_angles_group:
        The root HDF5 `Group` that contains the incident
        angle dataset specified by the pathname given by:

        * DatasetName.INCIDENT

    :param exiting_angles_group:
        The root HDF5 `Group` that contains the exiting
        angle dataset specified by the pathname given by:

        * DatasetName.EXITING

    :param shadow_masks_group:
        The root HDF5 `Group` that contains the combined shadow
        masks; self shadow, cast shadow (solar),
        cast shadow (satellite), dataset specified by the pathname
        given by:

        * DatasetName.COMBINED_SHADOW

    :param ancillary_group:
        The root HDF5 `Group` that contains the Isotropic (iso),
        RossThick (vol), and LiSparseR (geo) BRDF scalar parameters.
        The dataset pathnames are given by:

        * DatasetName.BRDF_FMT

    :param rori:
        Threshold for terrain correction. Fuqin to document.

    :param out_group:
        A writeable HDF5 `Group` object.

        The dataset names will be given by the format string detailed
        by:

        * DatasetName.REFLECTANCE_FMT

        The reflectance products are:

        * lambertian
        * nbar (BRDF corrected reflectance)
        * nbart (BRDF + terrain illumination corrected reflectance)

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

    :param normalized_solar_zenith:
        A float value type to normalize reflectance to a particular angle.

    :param esun
        A float value type. A solar irradiance normal to atmosphere
        in unit of W/sq cm/sr/nm.

    :return:
        An opened `h5py.File` object, that is either in-memory using the
        `core` driver, or on disk.
    """
    geobox = acquisition.gridded_geo_box()
    bn = acquisition.band_name

    dname_fmt = DatasetName.INTERPOLATION_FMT.value
    fv_dataset = interpolation_group[
        dname_fmt.format(coefficient=AC.FV.value, band_name=bn)
    ]
    fs_dataset = interpolation_group[
        dname_fmt.format(coefficient=AC.FS.value, band_name=bn)
    ]
    b_dataset = interpolation_group[
        dname_fmt.format(coefficient=AC.B.value, band_name=bn)
    ]
    s_dataset = interpolation_group[
        dname_fmt.format(coefficient=AC.S.value, band_name=bn)
    ]
    a_dataset = interpolation_group[
        dname_fmt.format(coefficient=AC.A.value, band_name=bn)
    ]
    dir_dataset = interpolation_group[
        dname_fmt.format(coefficient=AC.DIR.value, band_name=bn)
    ]
    dif_dataset = interpolation_group[
        dname_fmt.format(coefficient=AC.DIF.value, band_name=bn)
    ]
    ts_dataset = interpolation_group[
        dname_fmt.format(coefficient=AC.TS.value, band_name=bn)
    ]
    solar_zenith_dset = satellite_solar_group[DatasetName.SOLAR_ZENITH.value]
    solar_azimuth_dset = satellite_solar_group[DatasetName.SOLAR_AZIMUTH.value]
    satellite_v_dset = satellite_solar_group[DatasetName.SATELLITE_VIEW.value]
    relative_a_dset = satellite_solar_group[DatasetName.RELATIVE_AZIMUTH.value]
    slope_dataset = slope_aspect_group[DatasetName.SLOPE.value]
    aspect_dataset = slope_aspect_group[DatasetName.ASPECT.value]
    relative_s_dset = relative_slope_group[DatasetName.RELATIVE_SLOPE.value]
    incident_angle_dataset = incident_angles_group[DatasetName.INCIDENT.value]
    exiting_angle_dataset = exiting_angles_group[DatasetName.EXITING.value]
    shadow_dataset = shadow_masks_group[DatasetName.COMBINED_SHADOW.value]

    dname_fmt = DatasetName.BRDF_FMT.value
    dname = dname_fmt.format(
        band_name=bn, parameter=BrdfDirectionalParameters.ALPHA_1.value
    )
    brdf_alpha1 = ancillary_group[dname][()]

    dname = dname_fmt.format(
        band_name=bn, parameter=BrdfDirectionalParameters.ALPHA_2.value
    )
    brdf_alpha2 = ancillary_group[dname][()]

    assert out_group is not None
    fid = out_group

    if GroupName.STANDARD_GROUP.value not in fid:
        fid.create_group(GroupName.STANDARD_GROUP.value)

    kwargs = compression.settings(filter_opts, chunks=acquisition.tile_size)
    grp = fid[GroupName.STANDARD_GROUP.value]
    kwargs["shape"] = (acquisition.lines, acquisition.samples)
    kwargs["fillvalue"] = NO_DATA_VALUE
    kwargs["dtype"] = "int16"

    # create the datasets
    dname_fmt = DatasetName.REFLECTANCE_FMT.value
    dname = dname_fmt.format(product=AP.LAMBERTIAN.value, band_name=bn)
    lmbrt_dset = grp.create_dataset(dname, **kwargs)

    dname = dname_fmt.format(product=AP.NBAR.value, band_name=bn)
    nbar_dset = grp.create_dataset(dname, **kwargs)

    dname = dname_fmt.format(product=AP.NBART.value, band_name=bn)
    nbart_dset = grp.create_dataset(dname, **kwargs)

    # attach some attributes to the image datasets
    attrs = {
        "crs_wkt": geobox.crs.ExportToWkt(),
        "geotransform": geobox.transform.to_gdal(),
        "no_data_value": kwargs["fillvalue"],
        "rori_threshold_setting": rori,
        "platform_id": acquisition.platform_id,
        "sensor_id": acquisition.sensor_id,
        "band_id": acquisition.band_id,
        "band_name": bn,
        "alias": acquisition.alias,
    }

    desc = "Contains the lambertian reflectance data scaled by 10000."
    attrs["description"] = desc
    attach_image_attributes(lmbrt_dset, attrs)

    desc = "Contains the brdf corrected reflectance data scaled by 10000."
    attrs["description"] = desc
    attach_image_attributes(nbar_dset, attrs)

    desc = (
        "Contains the brdf and terrain corrected reflectance data scaled " "by 10000."
    )
    attrs["description"] = desc
    attach_image_attributes(nbart_dset, attrs)

    # process by tile
    for tile in acquisition.tiles():
        # tile indices
        idx = (slice(tile[0][0], tile[0][1]), slice(tile[1][0], tile[1][1]))

        # define some static arguments
        acq_args = {"window": tile, "out_no_data": NO_DATA_VALUE, "esun": esun}
        f32_args = {"dtype": np.float32, "transpose": True}

        # Read the data corresponding to the current tile for all dataset
        # Convert the datatype if required and transpose
        band_data = as_array(acquisition.radiance_data(**acq_args), **f32_args)

        if np.all(band_data == NO_DATA_VALUE):
            lmbrt_dset[idx] = NO_DATA_VALUE
            nbar_dset[idx] = NO_DATA_VALUE
            nbart_dset[idx] = NO_DATA_VALUE
            continue

        shadow = as_array(shadow_dataset[idx], np.int8, transpose=True)
        solar_zenith = as_array(solar_zenith_dset[idx], **f32_args)
        solar_azimuth = as_array(solar_azimuth_dset[idx], **f32_args)
        satellite_view = as_array(satellite_v_dset[idx], **f32_args)
        relative_angle = as_array(relative_a_dset[idx], **f32_args)
        slope = as_array(slope_dataset[idx], **f32_args)
        aspect = as_array(aspect_dataset[idx], **f32_args)
        incident_angle = as_array(incident_angle_dataset[idx], **f32_args)
        exiting_angle = as_array(exiting_angle_dataset[idx], **f32_args)
        relative_slope = as_array(relative_s_dset[idx], **f32_args)
        a_mod = as_array(a_dataset[idx], **f32_args)
        b_mod = as_array(b_dataset[idx], **f32_args)
        s_mod = as_array(s_dataset[idx], **f32_args)
        fs = as_array(fs_dataset[idx], **f32_args)
        fv = as_array(fv_dataset[idx], **f32_args)
        ts = as_array(ts_dataset[idx], **f32_args)
        direct = as_array(dir_dataset[idx], **f32_args)
        diffuse = as_array(dif_dataset[idx], **f32_args)

        # Allocate the output arrays
        xsize, ysize = band_data.shape  # band_data has been transposed
        ref_lm = np.zeros((ysize, xsize), dtype="int16")
        ref_brdf = np.zeros((ysize, xsize), dtype="int16")
        ref_terrain = np.zeros((ysize, xsize), dtype="int16")

        # Allocate the work arrays (single row of data)
        ref_lm_work = np.zeros(xsize, dtype="float32")
        ref_brdf_work = np.zeros(xsize, dtype="float32")
        ref_terrain_work = np.zeros(xsize, dtype="float32")

        # Run terrain correction
        reflectance_python(
            xsize,
            ysize,
            rori,
            brdf_alpha1,
            brdf_alpha2,
            acquisition.reflectance_adjustment,
            kwargs["fillvalue"],
            band_data,
            shadow,
            solar_zenith,
            solar_azimuth,
            satellite_view,
            relative_angle,
            slope,
            aspect,
            incident_angle,
            exiting_angle,
            relative_slope,
            a_mod,
            b_mod,
            s_mod,
            fs,
            fv,
            ts,
            direct,
            diffuse,
            ref_lm_work,
            ref_brdf_work,
            ref_terrain_work,
            ref_lm.transpose(),
            ref_brdf.transpose(),
            ref_terrain.transpose(),
            normalized_solar_zenith,
        )

        # Write the current tile to disk
        lmbrt_dset[idx] = ref_lm
        nbar_dset[idx] = ref_brdf
        nbart_dset[idx] = ref_terrain

    # close any still opened files, arrays etc associated with the acquisition
    acquisition.close()
