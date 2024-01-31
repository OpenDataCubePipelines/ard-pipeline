#!/usr/bin/env python


import json
import tempfile
from os.path import join as pjoin

import h5py

from wagl.acquisition import acquisitions
from wagl.ancillary import collect_ancillary
from wagl.constants import (
    ALBEDO_FMT,
    POINT_ALBEDO_FMT,
    POINT_FMT,
    Albedos,
    AtmosphericCoefficients,
    BandType,
    DatasetName,
    GroupName,
    Workflow,
)
from wagl.dsm import get_dsm
from wagl.hdf5 import H5CompressionFilter, read_h5_table
from wagl.incident_exiting_angles import (
    exiting_angles,
    incident_angles,
    relative_azimuth_slope,
)
from wagl.interpolation import interpolate
from wagl.logs import STATUS_LOGGER
from wagl.longitude_latitude_arrays import create_lon_lat_grids
from wagl.metadata import create_ard_yaml
from wagl.modtran import (
    JsonEncoder,
    calculate_coefficients,
    format_json,
    prepare_modtran,
    run_modtran,
)
from wagl.reflectance import calculate_reflectance
from wagl.satellite_solar_angles import calculate_angles
from wagl.slope_aspect import slope_aspect_arrays
from wagl.temperature import surface_brightness_temperature
from wagl.terrain_shadow_masks import (
    calculate_cast_shadow,
    combine_shadow_masks,
    self_shadow,
)


# pylint disable=too-many-arguments
def card4l(
    level1,
    granule,
    workflow,
    vertices,
    method,
    tle_path,
    aerosol,
    brdf,
    ozone_path,
    water_vapour,
    dem_path,
    srtm_pathname,
    cop_pathname,
    invariant_fname,
    modtran_exe,
    out_fname,
    ecmwf_path=None,
    rori=0.52,
    buffer_distance=8000,
    compression=H5CompressionFilter.LZF,
    filter_opts=None,
    h5_driver=None,
    acq_parser_hint=None,
    normalized_solar_zenith=45.0,
):
    """CEOS Analysis Ready Data for Land.
    A workflow for producing standardised products that meet the
    CARD4L specification.

    :param level1:
        A string containing the full file pathname to the level1
        dataset.

    :param granule:
        A string containing the granule id to process.

    :param workflow:
        An enum from wagl.constants.Workflow representing which
        workflow workflow to run.

    :param vertices:
        An integer 2-tuple indicating the number of rows and columns
        of sample-locations ("coordinator") to produce.
        The vertex columns should be an odd number.

    :param method:
        An enum from wagl.constants.Method representing the
        interpolation method to use during the interpolation
        of the atmospheric coefficients.

    :param tle_path:
        A string containing the full file pathname to the directory
        containing the two line element datasets.

    :param aerosol:
        A string containing the full file pathname to the HDF5 file
        containing the aerosol data.

    :param brdf:
        A dict containing either user-supplied BRDF values, or the
        full file pathname to the directory containing the BRDF data
        and the decadal averaged BRDF data used for acquisitions
        prior to TERRA/AQUA satellite operations.

    :param ozone_path:
        A string containing the full file pathname to the directory
        containing the ozone datasets.

    :param water_vapour:
        A string containing the full file pathname to the directory
        containing the water vapour datasets.

    :param dem_path:
        A string containing the full file pathname to the directory
        containing the reduced resolution DEM.

    :param srtm_pathname:
        A string pathname of the SRTM DSM with a ':' to seperate the
        filename from the import HDF5 dataset name.

    :param cop_pathname:
        A string pathname of the mosaiced Copernicus 30m DEM .tif file.

    :param invariant_fname:
        A string containing the full file pathname to the image file
        containing the invariant geo-potential data for use within
        the SBT process.

    :param modtran_exe:
        A string containing the full file pathname to the MODTRAN
        executable.

    :param out_fname:
        A string containing the full file pathname that will contain
        the output data from the data standardisation process.
        executable.

    :param ecmwf_path:
        A string containing the full file pathname to the directory
        containing the data from the European Centre for Medium Weather
        Forcast, for use within the SBT process.

    :param rori:
        A floating point value for surface reflectance adjustment.
        TODO Fuqin to add additional documentation for this parameter.
        Default is 0.52.

    :param buffer_distance:
        A number representing the desired distance (in the same
        units as the acquisition) in which to calculate the extra
        number of pixels required to buffer an image.
        Default is 8000, which for an acquisition using metres would
        equate to 8000 metres.

    :param compression:
        An enum from hdf5.compression.H5CompressionFilter representing
        the desired compression filter to use for writing H5 IMAGE and
        TABLE class datasets to disk.
        Default is H5CompressionFilter.LZF.

    :param filter_opts:
        A dict containing any additional keyword arguments when
        generating the configuration for the given compression Filter.
        Default is None.

    :param h5_driver:
        The specific HDF5 file driver to use when creating the output
        HDF5 file.
        See http://docs.h5py.org/en/latest/high/file.html#file-drivers
        for more details.
        Default is None; which writes direct to disk using the
        appropriate driver for the underlying OS.

    :param acq_parser_hint:
        A string containing any hints to provide the acquisitions
        loader with.

    :param normalized_solar_zenith:
        Solar zenith angle to normalize for (in degrees). Default is 45 degrees.
    """

    container = acquisitions(level1, hint=acq_parser_hint)

    # TODO: pass through an acquisitions container rather than pathname
    with h5py.File(out_fname, "w", driver=h5_driver) as fid:
        fid.attrs["level1_uri"] = level1

        # granule root group
        root = fid.create_group(granule)

        stash_oa_bands(
            root,
            container,
            granule,
            tle_path,
            workflow,
            srtm_pathname,
            cop_pathname,
            buffer_distance,
            compression,
            filter_opts,
        )

        stash_ancillary(
            root,
            container,
            granule,
            aerosol,
            water_vapour,
            ozone_path,
            dem_path,
            brdf,
            ecmwf_path,
            invariant_fname,
            vertices,
            compression,
            filter_opts,
        )

        stash_atmospherics(
            root,
            container,
            granule,
            workflow,
            vertices,
            modtran_exe,
            compression,
            filter_opts,
        )

        stash_interpolation(
            root,
            container,
            granule,
            workflow,
            method,
            compression,
            filter_opts,
        )

        stash_reflectance(
            root,
            container,
            granule,
            workflow,
            rori,
            normalized_solar_zenith,
            compression,
            filter_opts,
        )

        stash_metadata(
            root,
            container,
            granule,
            workflow,
            vertices,
            buffer_distance,
            method,
            rori,
            normalized_solar_zenith,
        )


def stash_oa_bands(
    granule_root,
    container,
    granule,
    tle_path,
    workflow,
    srtm_pathname,
    cop_pathname,
    buffer_distance,
    compression,
    filter_opts,
):
    for grp_name in container.supported_groups:
        log = STATUS_LOGGER.bind(
            level1=container.label, granule=granule, granule_group=grp_name
        )

        # root group for a given granule and resolution group
        root = granule_root.create_group(grp_name)
        acqs = container.get_acquisitions(granule=granule, group=grp_name)

        # include the resolution as a group attribute
        root.attrs["resolution"] = acqs[0].resolution

        # longitude and latitude
        log.info("Latitude-Longitude")
        create_lon_lat_grids(acqs[0], root, compression, filter_opts)

        # satellite and solar angles
        log.info("Satellite-Solar-Angles")
        calculate_angles(
            acqs[0],
            root[GroupName.LON_LAT_GROUP.value],
            root,
            compression,
            filter_opts,
            tle_path,
        )

        if workflow in (Workflow.STANDARD, Workflow.NBAR):
            # DEM
            log.info("DEM-retriveal")
            get_dsm(
                acqs[0],
                srtm_pathname,
                cop_pathname,
                buffer_distance,
                root,
                compression,
                filter_opts
            )

            # slope & aspect
            log.info("Slope-Aspect")
            slope_aspect_arrays(
                acqs[0],
                root[GroupName.ELEVATION_GROUP.value],
                buffer_distance,
                root,
                compression,
                filter_opts,
            )

            # incident angles
            log.info("Incident-Angles")
            incident_angles(
                root[GroupName.SAT_SOL_GROUP.value],
                root[GroupName.SLP_ASP_GROUP.value],
                root,
                compression,
                filter_opts,
            )

            # exiting angles
            log.info("Exiting-Angles")
            exiting_angles(
                root[GroupName.SAT_SOL_GROUP.value],
                root[GroupName.SLP_ASP_GROUP.value],
                root,
                compression,
                filter_opts,
            )

            # relative azimuth slope
            log.info("Relative-Azimuth-Angles")
            incident_group_name = GroupName.INCIDENT_GROUP.value
            exiting_group_name = GroupName.EXITING_GROUP.value
            relative_azimuth_slope(
                root[incident_group_name],
                root[exiting_group_name],
                root,
                compression,
                filter_opts,
            )

            # self shadow
            log.info("Self-Shadow")
            self_shadow(
                root[incident_group_name],
                root[exiting_group_name],
                root,
                compression,
                filter_opts,
            )

            # cast shadow solar source direction
            log.info("Cast-Shadow-Solar-Direction")
            dsm_group_name = GroupName.ELEVATION_GROUP.value
            calculate_cast_shadow(
                acqs[0],
                root[dsm_group_name],
                root[GroupName.SAT_SOL_GROUP.value],
                buffer_distance,
                root,
                compression,
                filter_opts,
            )

            # cast shadow satellite source direction
            log.info("Cast-Shadow-Satellite-Direction")
            calculate_cast_shadow(
                acqs[0],
                root[dsm_group_name],
                root[GroupName.SAT_SOL_GROUP.value],
                buffer_distance,
                root,
                compression,
                filter_opts,
                False,
            )

            # combined shadow masks
            log.info("Combined-Shadow")
            combine_shadow_masks(
                root[GroupName.SHADOW_GROUP.value],
                root[GroupName.SHADOW_GROUP.value],
                root[GroupName.SHADOW_GROUP.value],
                root,
                compression,
                filter_opts,
            )


def stash_ancillary(
    root,
    container,
    granule,
    aerosol,
    water_vapour,
    ozone_path,
    dem_path,
    brdf,
    ecmwf_path,
    invariant_fname,
    vertices,
    compression,
    filter_opts,
):
    # nbar and sbt ancillary
    log = STATUS_LOGGER.bind(
        level1=container.label, granule=granule, granule_group=None
    )

    # get the highest resolution group containing supported bands
    acqs, grp_name = container.get_highest_resolution(granule=granule)

    grn_con = container.get_granule(granule=granule, container=True)
    res_group = root[grp_name]

    log.info("Ancillary-Retrieval")
    nbar_paths = {
        "aerosol_dict": aerosol,
        "water_vapour_dict": water_vapour,
        "ozone_path": ozone_path,
        "dem_path": dem_path,
        "brdf_dict": brdf,
    }
    collect_ancillary(
        grn_con,
        res_group[GroupName.SAT_SOL_GROUP.value],
        nbar_paths,
        ecmwf_path,
        invariant_fname,
        vertices,
        root,
        compression,
        filter_opts,
    )


def stash_atmospherics(
    root,
    container,
    granule,
    workflow,
    vertices,
    modtran_exe,
    compression,
    filter_opts,
):
    log = STATUS_LOGGER.bind(
        level1=container.label, granule=granule, granule_group=None
    )

    # get the highest resolution group containing supported bands
    acqs, grp_name = container.get_highest_resolution(granule=granule)
    res_group = root[grp_name]

    # atmospherics
    log.info("Atmospherics")

    ancillary_group = root[GroupName.ANCILLARY_GROUP.value]

    # satellite/solar angles and lon/lat for a resolution group
    sat_sol_grp = res_group[GroupName.SAT_SOL_GROUP.value]
    lon_lat_grp = res_group[GroupName.LON_LAT_GROUP.value]

    # TODO: supported acqs in different groups pointing to different response funcs
    json_data, _ = format_json(
        acqs, ancillary_group, sat_sol_grp, lon_lat_grp, workflow, root
    )

    # atmospheric inputs group
    inputs_grp = root[GroupName.ATMOSPHERIC_INPUTS_GRP.value]

    json_fmt = pjoin(POINT_FMT, ALBEDO_FMT, "".join([POINT_ALBEDO_FMT, ".json"]))
    nvertices = vertices[0] * vertices[1]

    # radiative transfer for each point and albedo
    for key in json_data:
        point, albedo = key

        log.info("Radiative-Transfer", point=point, albedo=albedo.value)

        with tempfile.TemporaryDirectory() as tmpdir:
            prepare_modtran(acqs, point, [albedo], tmpdir)

            point_dir = pjoin(tmpdir, POINT_FMT.format(p=point))
            workdir = pjoin(point_dir, ALBEDO_FMT.format(a=albedo.value))

            json_mod_infile = pjoin(tmpdir, json_fmt.format(p=point, a=albedo.value))

            with open(json_mod_infile, "w") as src:
                json_dict = json_data[key]

                if albedo == Albedos.ALBEDO_TH:
                    json_dict["MODTRAN"][0]["MODTRANINPUT"]["SPECTRAL"]["FILTNM"] = (
                        "{}/{}".format(
                            workdir,
                            json_dict["MODTRAN"][0]["MODTRANINPUT"]["SPECTRAL"][
                                "FILTNM"
                            ],
                        )
                    )
                    json_dict["MODTRAN"][1]["MODTRANINPUT"]["SPECTRAL"]["FILTNM"] = (
                        "{}/{}".format(
                            workdir,
                            json_dict["MODTRAN"][1]["MODTRANINPUT"]["SPECTRAL"][
                                "FILTNM"
                            ],
                        )
                    )

                else:
                    json_dict["MODTRAN"][0]["MODTRANINPUT"]["SPECTRAL"]["FILTNM"] = (
                        "{}/{}".format(
                            workdir,
                            json_dict["MODTRAN"][0]["MODTRANINPUT"]["SPECTRAL"][
                                "FILTNM"
                            ],
                        )
                    )

                json.dump(json_dict, src, cls=JsonEncoder, indent=4)

            run_modtran(
                acqs,
                inputs_grp,
                workflow,
                nvertices,
                point,
                [albedo],
                modtran_exe,
                tmpdir,
                root,
                compression,
                filter_opts,
            )


def nbar_acquisitions(container, granule, grp_name):
    acqs = container.get_acquisitions(granule=granule, group=grp_name)
    return [acq for acq in acqs if acq.band_type == BandType.REFLECTIVE]


def sbt_acquisitions(container, granule, grp_name):
    acqs = container.get_acquisitions(granule=granule, group=grp_name)
    return [acq for acq in acqs if acq.band_type == BandType.THERMAL]


def relevant_acquisitions(container, granule, grp_name, workflow):
    band_acqs = []

    if workflow in (Workflow.STANDARD, Workflow.NBAR):
        band_acqs.extend(nbar_acquisitions(container, granule, grp_name))

    if workflow in (Workflow.STANDARD, Workflow.SBT):
        band_acqs.extend(sbt_acquisitions(container, granule, grp_name))

    return band_acqs


def stash_interpolation(
    root,
    container,
    granule,
    workflow,
    method,
    compression,
    filter_opts,
):
    # atmospheric coefficients
    log = STATUS_LOGGER.bind(
        level1=container.label, granule=granule, granule_group=None
    )
    log.info("Coefficients")
    ancillary_group = root[GroupName.ANCILLARY_GROUP.value]
    results_group = root[GroupName.ATMOSPHERIC_RESULTS_GRP.value]
    calculate_coefficients(results_group, root, compression, filter_opts)

    # interpolate coefficients
    for grp_name in container.supported_groups:
        log = STATUS_LOGGER.bind(
            level1=container.label, granule=granule, granule_group=grp_name
        )
        log.info("Interpolation")

        res_group = root[grp_name]
        sat_sol_grp = res_group[GroupName.SAT_SOL_GROUP.value]
        comp_grp = root[GroupName.COEFFICIENTS_GROUP.value]

        for coefficient in workflow.atmos_coefficients:
            if coefficient is AtmosphericCoefficients.ESUN:
                continue

            if coefficient in Workflow.NBAR.atmos_coefficients:
                band_acqs = nbar_acquisitions(container, granule, grp_name)
            else:
                band_acqs = sbt_acquisitions(container, granule, grp_name)

            for acq in band_acqs:
                log.info(
                    "Interpolate",
                    band_id=acq.band_id,
                    coefficient=coefficient.value,
                )
                interpolate(
                    acq,
                    coefficient,
                    ancillary_group,
                    sat_sol_grp,
                    comp_grp,
                    res_group,
                    compression,
                    filter_opts,
                    method,
                )


def get_esun_values(
    root,
    container,
    granule,
):
    esun_values = {}

    comp_grp = root[GroupName.COEFFICIENTS_GROUP.value]

    for grp_name in container.supported_groups:
        for acq in nbar_acquisitions(container, granule, grp_name):
            atmos_coefs = read_h5_table(comp_grp, DatasetName.NBAR_COEFFICIENTS.value)
            esun_values[acq.band_name] = (
                atmos_coefs[atmos_coefs.band_name == acq.band_name][
                    AtmosphericCoefficients.ESUN.value
                ]
            ).values[0]

    return esun_values


def stash_reflectance(
    root,
    container,
    granule,
    workflow,
    rori,
    normalized_solar_zenith,
    compression,
    filter_opts,
):
    ancillary_group = root[GroupName.ANCILLARY_GROUP.value]
    esun_values = get_esun_values(root, container, granule)

    for grp_name in container.supported_groups:
        log = STATUS_LOGGER.bind(
            level1=container.label, granule=granule, granule_group=grp_name
        )
        log.info("Reflectance")

        res_group = root[grp_name]
        sat_sol_grp = res_group[GroupName.SAT_SOL_GROUP.value]

        # standardised products
        band_acqs = relevant_acquisitions(container, granule, grp_name, workflow)

        for acq in band_acqs:
            interp_grp = res_group[GroupName.INTERP_GROUP.value]

            if acq.band_type == BandType.THERMAL:
                log.info("SBT", band_id=acq.band_id)
                surface_brightness_temperature(
                    acq, interp_grp, res_group, compression, filter_opts
                )
            else:
                slp_asp_grp = res_group[GroupName.SLP_ASP_GROUP.value]
                rel_slp_asp = res_group[GroupName.REL_SLP_GROUP.value]
                incident_grp = res_group[GroupName.INCIDENT_GROUP.value]
                exiting_grp = res_group[GroupName.EXITING_GROUP.value]
                shadow_grp = res_group[GroupName.SHADOW_GROUP.value]

                log.info("Surface-Reflectance", band_id=acq.band_id)
                calculate_reflectance(
                    acq,
                    interp_grp,
                    sat_sol_grp,
                    slp_asp_grp,
                    rel_slp_asp,
                    incident_grp,
                    exiting_grp,
                    shadow_grp,
                    ancillary_group,
                    rori,
                    res_group,
                    compression,
                    filter_opts,
                    normalized_solar_zenith,
                    esun_values[acq.band_name],
                )


def stash_metadata(
    root,
    container,
    granule,
    workflow,
    vertices,
    buffer_distance,
    method,
    rori,
    normalized_solar_zenith,
):
    ancillary_group = root[GroupName.ANCILLARY_GROUP.value]
    esun_values = get_esun_values(root, container, granule)

    # wagl parameters
    parameters = {
        "vertices": list(vertices),
        "method": method.value,
        "rori": rori,
        "buffer_distance": buffer_distance,
        "normalized_solar_zenith": normalized_solar_zenith,
        "esun": esun_values,
    }

    # metadata yaml's
    metadata = root.create_group(DatasetName.METADATA.value)
    create_ard_yaml(
        {
            grp_name: relevant_acquisitions(container, granule, grp_name, workflow)
            for grp_name in container.supported_groups
        },
        ancillary_group,
        metadata,
        parameters,
        workflow,
    )
