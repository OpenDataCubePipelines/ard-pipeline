#!/usr/bin/env python

"""Ancillary dataset retrieval and storage."""

import configparser
import datetime
import json
import os.path
from os.path import join as pjoin
from posixpath import join as ppjoin
from typing import Dict, List, Optional, Set, Tuple, TypedDict

import attr
import fiona
import h5py
import numpy as np
import pandas as pd
from geopandas import GeoSeries
from shapely import wkt
from shapely.geometry import Point, Polygon, shape

from wagl.acquisition import (
    Acquisition,
    AcquisitionsContainer,
)
from wagl.brdf import BrdfDict, BrdfMode, get_brdf_data
from wagl.constants import (
    POINT_FMT,
    AerosolTier,
    BandType,
    DatasetName,
    GroupName,
    OzoneTier,
    WaterVapourTier,
)
from wagl.data import get_pixel, get_pixel_from_raster
from wagl.hdf5 import (
    VLEN_STRING,
    H5CompressionFilter,
    attach_table_attributes,
    read_h5_table,
    write_dataframe,
    write_scalar,
)
from wagl.metadata import current_h5_metadata
from wagl.satellite_solar_angles import create_coordinator, create_vertices

#: A H5 file path and dataset name, separated by a colon.
#: h5_path, dataset_name = this.split(":")
PathWithDataset = str

LonLat = Tuple[float, float]


ECWMF_LEVELS = [
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
    50,
    70,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    775,
    800,
    825,
    850,
    875,
    900,
    925,
    950,
    975,
    1000,
]


class AncillaryError(Exception):
    """Specific error handle for ancillary retrieval."""


def get_4d_idx(day):
    """A small utility function for indexing into a 4D dataset
    represented as a 3D dataset.
    [month, level, y, x], where level contains 37 levels, and day
    contains 28, 29, 30 or 31 days.
    """
    start = 1 + 37 * (day - 1)
    stop = start + 37
    return list(range(start, stop, 1))


def kelvin_2_celcius(kelvin):
    """A small utility function for converting degrees Kelvin to
    degrees Celcius.
    """
    return kelvin - 273.15


def relative_humdity(surface_temp, dewpoint_temp, kelvin=True):
    """Calculates relative humidity given a surface temperature and
    dewpoint temperature.
    """
    if kelvin:
        surf_t = kelvin_2_celcius(surface_temp)
        dew_t = kelvin_2_celcius(dewpoint_temp)
    else:
        surf_t = surface_temp
        dew_t = dewpoint_temp

    rh = 100 * ((112.0 - 0.1 * surf_t + dew_t) / (112.0 + 0.9 * surf_t)) ** 8

    return rh


def check_interpolation_sample_geometry(container, group, grp_name):
    acqs = container.get_acquisitions(group=grp_name)
    acq = acqs[0]
    geobox = acq.gridded_geo_box()
    coord_read = read_h5_table(group, DatasetName.COORDINATOR.value)
    coord = np.zeros((coord_read.shape[0], 2), dtype="int")
    map_x = coord_read.map_x.values
    map_y = coord_read.map_y.values
    coord[:, 1], coord[:, 0] = (map_x, map_y) * ~geobox.transform

    unique_coords = {(coord[i, 0], coord[i, 1]) for i in range(coord.shape[0])}

    return coord.shape[0] == len(unique_coords)


def default_interpolation_grid(acquisition, vertices, boxline_dataset):
    geobox = acquisition.gridded_geo_box()
    rows, cols = geobox.shape
    istart = boxline_dataset["start_index"]
    iend = boxline_dataset["end_index"]
    nvertices = vertices[0] * vertices[1]
    locations = np.empty((vertices[0], vertices[1], 2), dtype="int64")
    grid_rows = list(
        np.linspace(0, rows - 1, vertices[0], endpoint=True, dtype="int32")
    )
    for ig, ir in enumerate(grid_rows):  # row indices for sample-grid & raster
        grid_line = np.linspace(istart[ir], iend[ir], vertices[1], endpoint=True)
        locations[ig, :, 0] = ir
        locations[ig, :, 1] = grid_line
    locations = locations.reshape(nvertices, 2)
    coordinator = create_coordinator(locations, geobox)
    return coordinator


class AerosolDict(TypedDict):
    # An optional, user-specified value.
    user: Optional[float]
    # HDF5 file path.
    pathname: str


class WaterVapourDict(TypedDict):
    # An optional, user-specified value.
    user: Optional[float]
    # The folder that water vapour files are found.
    # The files inside are expected of format: "pr_wtr.eatm.{year}.h5"
    pathname: str


class NbarPathsDict(TypedDict):
    aerosol_dict: AerosolDict
    water_vapour_dict: WaterVapourDict
    ozone_path: str
    dem_path: str
    cop_pathname: str
    brdf_dict: BrdfDict


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


def collect_ancillary(
    container: AcquisitionsContainer,
    satellite_solar_group,
    nbar_paths: NbarPathsDict,
    offshore_territory_boundary_path: str,
    sbt_path=None,
    invariant_fname=None,
    vertices=(3, 3),
    out_group=None,
    compression=H5CompressionFilter.LZF,
    filter_opts=None,
):
    """Collects the ancillary required for NBAR and optionally SBT.
    This could be better handled if using the `opendatacube` project
    to handle ancillary retrieval, rather than directory passing,
    and filename grepping.

    :param container:
        An instance of an `AcquisitionsContainer` object.
        The container should consist of a single Granule or None,
        only. Use `AcquisitionsContainer.get_granule` method prior to
        calling this function.

    :param satellite_solar_group:
        The root HDF5 `Group` that contains the solar zenith and
        solar azimuth datasets specified by the pathnames given by:

        * DatasetName.BOXLINE

    :param nbar_paths:
        A `dict` containing the ancillary pathnames required for
        retrieving the NBAR ancillary data. Required keys:

        * aerosol_data
        * water_vapour_data
        * ozone_path
        * dem_path
        * brdf_dict

    :param offshore_territory_boundary_path:
        A `str` to the path of a geometry file delineating the boundary
        the outside of which is considered "offshore"

    :param sbt_path:
        A `str` containing the base directory pointing to the
        ancillary products required for the SBT workflow.

    :param invariant_fname:
        A `str` containing the file path name to the invariant
        geopotential image file.

    :param vertices:
        An integer 2-tuple indicating the number of rows and columns
        of sample-locations ("coordinator") to produce.
        The vertex columns should be an odd number.
        Default is (3, 3).

    :param out_group:
        A writeable HDF5 `Group` object.

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

    # This requirement was stated in the docstring, let's just enforce it.
    if len(container.granules) > 1:
        raise RuntimeError(
            "Container should consist of a single Granule or None, only. "
            "Use `AcquisitionsContainer.get_granule` method prior to calling this function."
        )

    assert out_group is not None
    fid = out_group

    if filter_opts is None:
        filter_opts = {}

    kwargs = compression.config(**filter_opts).dataset_compression_kwargs()
    group = fid.create_group(GroupName.ANCILLARY_GROUP.value)

    acquisition = container.get_highest_resolution()[0][0]

    boxline_dataset = satellite_solar_group[DatasetName.BOXLINE.value][:]
    coordinator = create_vertices(acquisition, boxline_dataset, vertices)
    lonlats = zip(coordinator["longitude"], coordinator["latitude"])

    desc = (
        "Contains the row and column array coordinates used for the "
        "atmospheric calculations."
    )
    attrs = {"description": desc, "array_coordinate_offset": 0}
    kwargs = compression.config(**filter_opts).dataset_compression_kwargs()
    dset_name = DatasetName.COORDINATOR.value
    coord_dset = group.create_dataset(dset_name, data=coordinator, **kwargs)
    attach_table_attributes(coord_dset, title="Coordinator", attrs=attrs)

    # check if modtran interpolation points coincide
    if not all(
        check_interpolation_sample_geometry(container, group, grp_name)
        for grp_name in container.supported_groups
    ):
        coord_dset[:] = default_interpolation_grid(
            acquisition, vertices, boxline_dataset
        )
        attach_table_attributes(coord_dset, title="Coordinator", attrs=attrs)

    if sbt_path:
        collect_sbt_ancillary(
            acquisition,
            lonlats,
            sbt_path,
            invariant_fname,
            out_group=group,
            compression=compression,
            filter_opts=filter_opts,
        )

    collect_nbar_ancillary(
        container,
        out_group=group,
        offshore=is_offshore_territory(acquisition, offshore_territory_boundary_path),
        compression=compression,
        filter_opts=filter_opts,
        **nbar_paths,
    )


def collect_sbt_ancillary(
    acquisition,
    lonlats,
    ancillary_path,
    invariant_fname=None,
    out_group=None,
    compression=H5CompressionFilter.LZF,
    filter_opts=None,
):
    """Collects the ancillary data required for surface brightness
    temperature.

    :param acquisition:
        An instance of an `Acquisition` object.

    :param lonlats:
        A `list` of tuples containing (longitude, latitude) coordinates.

    :param ancillary_path:
        A `str` containing the directory pathname to the ECMWF
        ancillary data.

    :param invariant_fname:
        A `str` containing the file pathname to the invariant geopotential
        data.

    :param out_group:
        A writeable HDF5 `Group` object.

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
    assert out_group is not None
    fid = out_group

    fid.attrs["sbt-ancillary"] = True

    dt = acquisition.acquisition_datetime

    description = (
        "Combined Surface and Pressure Layer data retrieved from "
        "the ECWMF catalogue."
    )
    attrs = {"description": description, "Date used for querying ECWMF": dt}

    for i, lonlat in enumerate(lonlats):
        pnt = POINT_FMT.format(p=i)
        # get data located at the surface
        dew = ecwmf_dewpoint_temperature(ancillary_path, lonlat, dt)
        t2m = ecwmf_temperature_2metre(ancillary_path, lonlat, dt)
        sfc_prs = ecwmf_surface_pressure(ancillary_path, lonlat, dt)
        sfc_hgt = ecwmf_elevation(invariant_fname, lonlat)
        sfc_rh = relative_humdity(t2m[0], dew[0])

        # output the scalar data along with the attrs
        dname = ppjoin(pnt, DatasetName.DEWPOINT_TEMPERATURE.value)
        write_scalar(dew[0], dname, fid, dew[1])

        dname = ppjoin(pnt, DatasetName.TEMPERATURE_2M.value)
        write_scalar(t2m[0], dname, fid, t2m[1])

        dname = ppjoin(pnt, DatasetName.SURFACE_PRESSURE.value)
        write_scalar(sfc_prs[0], dname, fid, sfc_prs[1])

        dname = ppjoin(pnt, DatasetName.SURFACE_GEOPOTENTIAL.value)
        write_scalar(sfc_hgt[0], dname, fid, sfc_hgt[1])

        dname = ppjoin(pnt, DatasetName.SURFACE_RELATIVE_HUMIDITY.value)
        attrs = {"description": "Relative Humidity calculated at the surface"}
        write_scalar(sfc_rh, dname, fid, attrs)

        # get the data from each of the pressure levels (1 -> 1000 ISBL)
        gph = ecwmf_geo_potential(ancillary_path, lonlat, dt)
        tmp = ecwmf_temperature(ancillary_path, lonlat, dt)
        rh = ecwmf_relative_humidity(ancillary_path, lonlat, dt)

        dname = ppjoin(pnt, DatasetName.GEOPOTENTIAL.value)
        write_dataframe(
            gph[0], dname, fid, compression, attrs=gph[1], filter_opts=filter_opts
        )

        dname = ppjoin(pnt, DatasetName.TEMPERATURE.value)
        write_dataframe(
            tmp[0], dname, fid, compression, attrs=tmp[1], filter_opts=filter_opts
        )

        dname = ppjoin(pnt, DatasetName.RELATIVE_HUMIDITY.value)
        write_dataframe(
            rh[0], dname, fid, compression, attrs=rh[1], filter_opts=filter_opts
        )

        # combine the surface and higher pressure layers into a single array
        cols = ["GeoPotential_Height", "Pressure", "Temperature", "Relative_Humidity"]
        layers = pd.DataFrame(
            columns=cols, index=range(rh[0].shape[0]), dtype="float64"
        )

        layers["GeoPotential_Height"] = gph[0]["GeoPotential_Height"].values
        layers["Pressure"] = ECWMF_LEVELS[::-1]
        layers["Temperature"] = tmp[0]["Temperature"].values
        layers["Relative_Humidity"] = rh[0]["Relative_Humidity"].values

        # define the surface level
        df = pd.DataFrame(
            {
                "GeoPotential_Height": sfc_hgt[0],
                "Pressure": sfc_prs[0],
                "Temperature": kelvin_2_celcius(t2m[0]),
                "Relative_Humidity": sfc_rh,
            },
            index=[0],
        )

        # MODTRAN requires the height to be ascending
        # and the pressure to be descending
        wh = (layers["GeoPotential_Height"] > sfc_hgt[0]) & (
            layers["Pressure"] < sfc_prs[0].round()
        )
        df = df.append(layers[wh])
        df.reset_index(drop=True, inplace=True)

        dname = ppjoin(pnt, DatasetName.ATMOSPHERIC_PROFILE.value)
        write_dataframe(
            df, dname, fid, compression, attrs=attrs, filter_opts=filter_opts
        )

        fid[pnt].attrs["lonlat"] = lonlat


@attr.define
class AncillaryConfig:
    """
    The configuration settings for finding ancillary data used in data processing.
    """

    aerosol_dict: AerosolDict
    water_vapour_dict: WaterVapourDict
    dem_path: PathWithDataset
    brdf_dict: BrdfDict
    ozone_path: Optional[str] = None

    @classmethod
    def from_luigi(cls, luigi_config_path: Optional[str] = None):
        """
        Load ancillary config from luigi config file.

        If no path is given, it will try to load from the default luigi locations.

            /etc/luigi/luigi.cfg
            luigi.cfg
            $LUIGI_CONFIG_PATH

        (the latter is set by most DEA modules on load)
        """
        if not luigi_config_path:
            for path in (
                "/etc/luigi/luigi.cfg",
                "luigi.cfg",
                os.environ.get("LUIGI_CONFIG_PATH"),
            ):
                if path and os.path.exists(path):
                    luigi_config_path = path
                    break
            else:
                raise ValueError(
                    "No luigi config path given, and no default config found"
                )

        config = configparser.ConfigParser()
        config.read(luigi_config_path)

        def get_dict(field):
            try:
                return json.loads(config.get("DataStandardisation", field))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Config is not a valid json dict: DataStandardisation->{field!r}"
                ) from e

        return cls(
            aerosol_dict=get_dict("aerosol"),
            water_vapour_dict=get_dict("water_vapour"),
            ozone_path=config.get("DataStandardisation", "ozone_path"),
            dem_path=config.get("DataStandardisation", "dem_path"),
            brdf_dict=get_dict("brdf"),
        )


def find_needed_acquisition_ancillary(
    acquisition: Acquisition,
    config: AncillaryConfig,
    mode: Optional[BrdfMode] = None,
    offshore: bool = False,
) -> Tuple[Set[str], List[str]]:
    """
    Find which Ancillary Paths are needed to process this acquisition.

    Parameters:
    -----------
    container : AcquisitionsContainer
        Container object containing acquisitions.
    config : AncillaryConfig
        Config object containing ancillary paths.

    Returns:
    --------
        The tiers ("DEFINITIVE", "FALLBACK_DATASET") being used and the list of ancillary
        paths needed.

        The ancillary is found separately for ALPHA_1 and ALPHA_2, so
        in theory there could be two tiers. (?)

    """
    dem_file_path = config.dem_path.split(":")[0]

    paths = [
        config.aerosol_dict["pathname"],
        find_water_vapour_definitive_path(acquisition, config.water_vapour_dict),
        config.ozone_path,
        dem_file_path,
        config.brdf_dict["ocean_mask_path"]
        if not offshore
        else config.brdf_dict["extended_ocean_mask_path"],
    ]

    tiers: Set[str] = set()

    # This currently loads the H5 files. Maybe they're empty?
    params = get_brdf_data(acquisition, config.brdf_dict, mode=mode)
    for param, brdf_data in params.items():
        tiers.add(brdf_data["tier"])
        paths.extend(brdf_data["local_source_paths"])

    return tiers, paths


def collect_nbar_ancillary(
    container: AcquisitionsContainer,
    aerosol_dict: AerosolDict = None,
    water_vapour_dict: WaterVapourDict = None,
    ozone_path: Optional[str] = None,
    dem_path: PathWithDataset = None,
    cop_pathname: Optional[str] = None,
    brdf_dict: BrdfDict = None,
    offshore: bool = False,
    out_group=None,
    compression=H5CompressionFilter.LZF,
    filter_opts=None,
):
    """Collects the ancillary information required to create NBAR.

    :param container:
        An instance of an `AcquisitionsContainer` object.

    :param aerosol_dict:
        A `dict` defined as either of the following:

        * {'user': <value>}
        * {'pathname': <value>}

    :param water_vapour_dict:
        A `dict` defined as either of the following:

        * {'user': <value>}
        * {'pathname': <value>}

    :param ozone_path:
        A `str` containing the full file pathname to the directory
        containing the ozone data.

    :param dem_path:
        A `str` containing the full file pathname to the directory
        containing the digital elevation model data.

    :param brdf_dict:
        A `dict` defined as either of the following:

        * {'user': {<band-alias>: {'alpha_1': <value>, 'alpha_2': <value>}, ...}}
        * {'brdf_path': <path-to-BRDF>, 'brdf_fallback_path': <path-to-average-BRDF>}

    :param offshore:
        Whether the acquisition to be processed belongs to Australian offshore territories

    :param out_group:
        A writeable HDF5 `Group` object.

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

    :notes:
        The keywords compression and filter_opts aren't used as we no
        longer save the BRDF imagery. However, we may need to store
        tables in future, therefore they can remain until we know
        for sure they'll never be used.
    """
    assert out_group is not None
    fid = out_group

    acquisition = container.get_highest_resolution()[0][0]
    dt = acquisition.acquisition_datetime
    geobox = acquisition.gridded_geo_box()

    aerosol = get_aerosol_data(acquisition, aerosol_dict)
    write_scalar(aerosol[0], DatasetName.AEROSOL.value, fid, aerosol[1])

    wv = get_water_vapour(acquisition, water_vapour_dict)
    write_scalar(wv[0], DatasetName.WATER_VAPOUR.value, fid, wv[1])

    ozone = get_ozone_data(ozone_path, geobox.centre_lonlat, dt)
    write_scalar(ozone[0], DatasetName.OZONE.value, fid, ozone[1])

    if offshore:
        dsm_path = cop_pathname
    else:
        dsm_path = dem_path
    elev = get_elevation_data(geobox.centre_lonlat, dsm_path, offshore)
    write_scalar(elev[0], DatasetName.ELEVATION.value, fid, elev[1])

    # brdf
    dname_format = DatasetName.BRDF_FMT.value
    for group in container.groups:
        for acq in container.get_acquisitions(group=group):
            if acq.band_type is not BandType.REFLECTIVE:
                continue
            data = get_brdf_data(
                acq, brdf_dict, compression=compression, offshore=offshore
            )

            # output
            for param in data:
                dname = dname_format.format(
                    parameter=param.value, band_name=acq.band_name
                )
                brdf_value = data[param].pop("value")
                write_scalar(brdf_value, dname, fid, data[param])


def get_aerosol_data(
    acquisition: Acquisition, aerosol_dict: AerosolDict
) -> Tuple[float, Dict]:
    """Extract the aerosol value for an acquisition.
    The version 2 retrieves the data from a HDF5 file, and provides
    more control over how the data is selected geo-metrically.
    Better control over timedeltas.
    """
    aerosol_fname = aerosol_dict["pathname"]

    dt = acquisition.acquisition_datetime
    geobox = acquisition.gridded_geo_box()
    roi_poly = Polygon(
        [geobox.ul_lonlat, geobox.ur_lonlat, geobox.lr_lonlat, geobox.ll_lonlat]
    )

    descr = ["AATSR_PIX", "AATSR_CMP_YEAR_MONTH", "AATSR_CMP_MONTH"]
    names = ["ATSR_LF_%Y%m", "aot_mean_%b_%Y_All_Aerosols", "aot_mean_%b_All_Aerosols"]
    exts = ["/pix", "/cmp", "/cmp"]
    pathnames = [ppjoin(ext, dt.strftime(n)) for ext, n in zip(exts, names)]

    # temporary until we sort out a better default mechanism
    # how do we want to support default values, whilst still support provenance
    if "user" in aerosol_dict:
        tier = AerosolTier.USER
        metadata = {"id": np.array([], VLEN_STRING), "tier": tier.name}

        return aerosol_dict["user"], metadata

    data = None
    delta_tolerance = datetime.timedelta(days=0.5)
    with h5py.File(aerosol_fname, "r") as fid:
        for pathname, description in zip(pathnames, descr):
            tier = AerosolTier[description]
            if pathname in fid:
                df = read_h5_table(fid, pathname)
                aerosol_poly = wkt.loads(fid[pathname].attrs["extents"])

                if aerosol_poly.intersects(roi_poly):
                    if description == "AATSR_PIX":
                        abs_diff = (df["timestamp"] - dt).abs()
                        df = df[abs_diff < delta_tolerance]
                        df.reset_index(inplace=True, drop=True)

                    if df.shape[0] == 0:
                        continue

                    intersection = aerosol_poly.intersection(roi_poly)
                    pts = GeoSeries([Point(x, y) for x, y in zip(df["lon"], df["lat"])])
                    idx = pts.within(intersection)
                    data = df[idx]["aerosol"].mean()

                    if np.isfinite(data):
                        # ancillary metadata tracking
                        md = current_h5_metadata(fid, dataset_path=pathname)
                        metadata = {
                            "id": np.array([md["id"]], VLEN_STRING),
                            "tier": tier.name,
                        }

                        return data, metadata

    # default aerosol value
    data = 0.06
    metadata = {
        "id": np.array([], VLEN_STRING),
        "tier": AerosolTier.FALLBACK_DEFAULT.name,
    }

    return data, metadata


def get_elevation_data(lonlat: LonLat, pathname: PathWithDataset, offshore: bool):
    """Get elevation data for a scene.

    :param lon_lat:
        The latitude, longitude of the scene center.
    :type lon_lat:
        float (2-tuple)

    :pathname:
        The pathname of the DEM with a ':' to separate the
        dataset name.
    :type dem_dir:
        str
    """

    try:
        if not offshore:
            fname, dname = pathname.split(":")
            data, md_uuid = get_pixel(fname, dname, lonlat)
            metadata = {"id": np.array([md_uuid], VLEN_STRING)}
        else:
            data = get_pixel_from_raster(pathname, lonlat)
            metadata = {"id": np.array(["cop-30m-dem"], VLEN_STRING)}
        data = data * 0.001  # scale to correct units
    except ValueError:
        raise AncillaryError("No Elevation data")

    return data, metadata


def get_ozone_data(ozone_fname: str, lonlat: LonLat, acq_time: datetime.datetime):
    """Get ozone data for a scene. `lonlat` should be the (x,y) for the centre
    the scene.
    """
    dname = acq_time.strftime("%b").lower()

    try:
        data, md_uuid = get_pixel(ozone_fname, dname, lonlat)
        metadata = {
            "id": np.array([md_uuid], VLEN_STRING),
            "tier": OzoneTier.DEFINITIVE.name,
        }
    except IndexError:
        # Coords for expanded AOI in long lat
        coords = [[70.3, -56.7], [170.0, -56.7], [170.0, -7.8], [70.3, -7.8]]
        polygon = Polygon(coords)
        point = Point(lonlat)

        if polygon.contains(point):
            data = 0.275  # atm-cm or 275 Dobson Units (DU)
            metadata = {
                "id": np.array([], VLEN_STRING),
                "tier": OzoneTier.USER.name,
            }
        else:
            raise AncillaryError("No Ozone data")

    return data, metadata


def get_water_vapour(
    acquisition: Acquisition,
    water_vapour_dict: WaterVapourDict,
    scale_factor=0.1,
    tolerance=1,
):
    """Retrieve the water vapour value for an `acquisition` and the
    path for the water vapour ancillary data.
    """
    datafile = find_water_vapour_definitive_path(acquisition, water_vapour_dict)

    dt = acquisition.acquisition_datetime
    hour = dt.timetuple().tm_hour

    if "user" in water_vapour_dict:
        metadata = {"id": np.array([], VLEN_STRING), "tier": WaterVapourTier.USER.name}
        return water_vapour_dict["user"], metadata

    geobox = acquisition.gridded_geo_box()

    if os.path.isfile(datafile):
        with h5py.File(datafile, "r") as fid:
            index = read_h5_table(fid, "INDEX")

        # set the tolerance in days to search back in time
        max_tolerance = -datetime.timedelta(days=tolerance)

        # only look for observations that have occured in the past
        time_delta = index.timestamp - dt
        result = time_delta[
            (time_delta < datetime.timedelta()) & (time_delta > max_tolerance)
        ]

    if not os.path.isfile(datafile) or result.shape[0] == 0:
        if "fallback_dataset" not in water_vapour_dict:
            raise AncillaryError("No actual or fallback water vapour data.")

        tier = WaterVapourTier.FALLBACK_DATASET
        month = dt.strftime("%B-%d").upper()

        # closest previous observation
        # i.e. observations are at 0000, 0600, 1200, 1800
        # and an acquisition hour of 1700 will use the 1200 observation
        observations = np.array([0, 6, 12, 18])
        hr = observations[np.argmin(np.abs(hour - observations))]
        dataset_name = f"AVERAGE/{month}/{hr:02d}00"
        datafile = water_vapour_dict["fallback_dataset"]
    else:
        tier = WaterVapourTier.DEFINITIVE
        # get the index of the closest water vapour observation
        # which would be the maximum timedelta
        # as we're only dealing with negative timedelta's here
        idx = result.idxmax()
        record = index.iloc[idx]
        dataset_name = record.dataset_name

    if isinstance(dataset_name, bytes):
        dataset_name = dataset_name.decode("utf-8")

    try:
        data, md_uuid = get_pixel(datafile, dataset_name, geobox.centre_lonlat)
    except ValueError:
        # h5py raises a ValueError not an IndexError for out of bounds
        raise AncillaryError("No Water Vapour data")

    # the metadata from the original file says (Kg/m^2)
    # so multiply by 0.1 to get (g/cm^2)
    data = data * scale_factor
    metadata = {"id": np.array([md_uuid], VLEN_STRING), "tier": tier.name}

    return data, metadata


def find_water_vapour_definitive_path(
    acquisition: Acquisition, water_vapour_dict: Dict[str, str]
) -> str:
    dat = acquisition.acquisition_datetime
    year = dat.strftime("%Y")
    filename = f"pr_wtr.eatm.{year}.h5"
    water_vapour_path = water_vapour_dict["pathname"]
    datafile = pjoin(water_vapour_path, filename)
    return datafile


def ecwmf_elevation(datafile, lonlat):
    """Retrieve a pixel from the ECWMF invariant geo-potential
    dataset.
    Converts to Geo-Potential height in KM.
    2 metres is added to the result before returning.
    """
    # TODO; have swfo convert the files to HDF5
    raise AncillaryError("No Invariant Geo-Potential data")
    # try:
    #     data = get_pixel(datafile, lonlat) / 9.80665 / 1000.0 + 0.002
    # except IndexError:
    #     raise AncillaryError("No Invariant Geo-Potential data")

    # url = urlparse(datafile, scheme='file').geturl()

    # metadata = {'data_source': 'ECWMF Invariant Geo-Potential',
    #             'url': url}

    # # ancillary metadata tracking
    # md = extract_ancillary_metadata(datafile)
    # for key in md:
    #     metadata[key] = md[key]

    # return data, metadata


def ecwmf_temperature_2metre(input_path, lonlat, time):
    """Retrieve a pixel value from the ECWMF 2 metre Temperature
    collection.
    """
    # TODO; have swfo convert the files to HDF5
    raise AncillaryError("No ECWMF 2 metre Temperature data")
    # product = DatasetName.TEMPERATURE_2M.value.lower()
    # search = pjoin(input_path, DatasetName.ECMWF_PATH_FMT.value)
    # files = glob.glob(search.format(product=product, year=time.year))
    # data = None
    # required_ymd = datetime.datetime(time.year, time.month, time.day)
    # for f in files:
    #     url = urlparse(f, scheme='file').geturl()
    #     ymd = splitext(basename(f))[0].split('_')[1]
    #     ancillary_ymd = datetime.datetime.strptime(ymd, '%Y-%m-%d')
    #     if ancillary_ymd == required_ymd:
    #         data = get_pixel(f, lonlat)

    #         metadata = {'data_source': 'ECWMF 2 metre Temperature',
    #                     'url': url,
    #                     'query_date': time}

    #         # ancillary metadata tracking
    #         md = extract_ancillary_metadata(f)
    #         for key in md:
    #             metadata[key] = md[key]

    #         return data, metadata

    # if data is None:
    #     raise AncillaryError("No ECWMF 2 metre Temperature data")


def ecwmf_dewpoint_temperature(input_path, lonlat, time):
    """Retrieve a pixel value from the ECWMF 2 metre Dewpoint
    Temperature collection.
    """
    # TODO; have swfo convert the files to HDF5
    raise AncillaryError("No ECWMF 2 metre Dewpoint Temperature data")
    # product = DatasetName.DEWPOINT_TEMPERATURE.value.lower()
    # search = pjoin(input_path, DatasetName.ECMWF_PATH_FMT.value)
    # files = glob.glob(search.format(product=product, year=time.year))
    # data = None
    # required_ymd = datetime.datetime(time.year, time.month, time.day)
    # for f in files:
    #     url = urlparse(f, scheme='file').geturl()
    #     ymd = splitext(basename(f))[0].split('_')[1]
    #     ancillary_ymd = datetime.datetime.strptime(ymd, '%Y-%m-%d')
    #     if ancillary_ymd == required_ymd:
    #         data = get_pixel(f, lonlat)

    #         metadata = {'data_source': 'ECWMF 2 metre Dewpoint Temperature ',
    #                     'url': url,
    #                     'query_date': time}

    #         # ancillary metadata tracking
    #         md = extract_ancillary_metadata(f)
    #         for key in md:
    #             metadata[key] = md[key]

    #         return data, metadata

    # if data is None:
    #     raise AncillaryError("No ECWMF 2 metre Dewpoint Temperature data")


def ecwmf_surface_pressure(input_path, lonlat, time):
    """Retrieve a pixel value from the ECWMF Surface Pressure
    collection.
    Scales the result by 100 before returning.
    """
    # TODO; have swfo convert the files to HDF5
    raise AncillaryError("No ECWMF Surface Pressure data")
    # product = DatasetName.SURFACE_PRESSURE.value.lower()
    # search = pjoin(input_path, DatasetName.ECMWF_PATH_FMT.value)
    # files = glob.glob(search.format(product=product, year=time.year))
    # data = None
    # required_ymd = datetime.datetime(time.year, time.month, time.day)
    # for f in files:
    #     url = urlparse(f, scheme='file').geturl()
    #     ymd = splitext(basename(f))[0].split('_')[1]
    #     ancillary_ymd = datetime.datetime.strptime(ymd, '%Y-%m-%d')
    #     if ancillary_ymd == required_ymd:
    #         data = get_pixel(f, lonlat) / 100.0

    #         metadata = {'data_source': 'ECWMF Surface Pressure',
    #                     'url': url,
    #                     'query_date': time}

    #         # ancillary metadata tracking
    #         md = extract_ancillary_metadata(f)
    #         for key in md:
    #             metadata[key] = md[key]

    #         return data, metadata

    # if data is None:
    #     raise AncillaryError("No ECWMF Surface Pressure data")


def ecwmf_water_vapour(input_path, lonlat, time):
    """Retrieve a pixel value from the ECWMF Total Column Water Vapour
    collection.
    """
    # TODO; have swfo convert the files to HDF5
    raise AncillaryError("No ECWMF Total Column Water Vapour data")
    # product = DatasetName.WATER_VAPOUR.value.lower()
    # search = pjoin(input_path, DatasetName.ECMWF_PATH_FMT.value)
    # files = glob.glob(search.format(product=product, year=time.year))
    # data = None
    # required_ymd = datetime.datetime(time.year, time.month, time.day)
    # for f in files:
    #     url = urlparse(f, scheme='file').geturl()
    #     ymd = splitext(basename(f))[0].split('_')[1]
    #     ancillary_ymd = datetime.datetime.strptime(ymd, '%Y-%m-%d')
    #     if ancillary_ymd == required_ymd:
    #         data = get_pixel(f, lonlat)

    #         metadata = {'data_source': 'ECWMF Total Column Water Vapour',
    #                     'url': url,
    #                     'query_date': time}

    #         # ancillary metadata tracking
    #         md = extract_ancillary_metadata(f)
    #         for key in md:
    #             metadata[key] = md[key]

    #         return data, metadata

    # if data is None:
    #     raise AncillaryError("No ECWMF Total Column Water Vapour data")


def ecwmf_temperature(input_path, lonlat, time):
    """Retrieve a pixel value from the ECWMF Temperature collection
    across 37 height pressure levels, for a given longitude,
    latitude and time.

    Reverses the order of elements
    (1000 -> 1 mb, rather than 1 -> 1000 mb) before returning.
    """
    # TODO; have swfo convert the files to HDF5
    raise AncillaryError("No ECWMF Temperature profile data")
    # product = DatasetName.TEMPERATURE.value.lower()
    # search = pjoin(input_path, DatasetName.ECMWF_PATH_FMT.value)
    # files = glob.glob(search.format(product=product, year=time.year))
    # data = None
    # required_ymd = datetime.datetime(time.year, time.month, time.day)
    # for f in files:
    #     url = urlparse(f, scheme='file').geturl()
    #     ymd = splitext(basename(f))[0].split('_')[1]
    #     ancillary_ymd = datetime.datetime.strptime(ymd, '%Y-%m-%d')
    #     if ancillary_ymd == required_ymd:
    #         bands = list(range(1, 38))
    #         data = get_pixel(f, lonlat, bands)[::-1]

    #         metadata = {'data_source': 'ECWMF Temperature',
    #                     'url': url,
    #                     'query_date': time}

    #         # ancillary metadata tracking
    #         md = extract_ancillary_metadata(f)
    #         for key in md:
    #             metadata[key] = md[key]

    #         # internal file metadata (and reverse the ordering)
    #         df = read_metadata_tags(f, bands).iloc[::-1]
    #         df.insert(0, 'Temperature', data)

    #         return df, metadata

    # if data is None:
    #     raise AncillaryError("No ECWMF Temperature profile data")


def ecwmf_geo_potential(input_path, lonlat, time):
    """Retrieve a pixel value from the ECWMF Geo-Potential collection
    across 37 height pressure levels, for a given longitude,
    latitude and time.

    Converts to geo-potential height in KM, and reverses the order of
    the elements (1000 -> 1 mb, rather than 1 -> 1000 mb) before
    returning.
    """
    # TODO; have swfo convert the files to HDF5
    raise AncillaryError("No ECWMF Geo-Potential profile data")
    # product = DatasetName.GEOPOTENTIAL.value.lower()
    # search = pjoin(input_path, DatasetName.ECMWF_PATH_FMT.value)
    # files = glob.glob(search.format(product=product, year=time.year))
    # data = None
    # required_ymd = datetime.datetime(time.year, time.month, time.day)
    # for f in files:
    #     url = urlparse(f, scheme='file').geturl()
    #     ymd = splitext(basename(f))[0].split('_')[1]
    #     ancillary_ymd = datetime.datetime.strptime(ymd, '%Y-%m-%d')
    #     if ancillary_ymd == required_ymd:
    #         bands = list(range(1, 38))
    #         data = get_pixel(f, lonlat, bands)[::-1]
    #         scaled_data = data / 9.80665 / 1000.0

    #         metadata = {'data_source': 'ECWMF Geo-Potential',
    #                     'url': url,
    #                     'query_date': time}

    #         # ancillary metadata tracking
    #         md = extract_ancillary_metadata(f)
    #         for key in md:
    #             metadata[key] = md[key]

    #         # internal file metadata (and reverse the ordering)
    #         df = read_metadata_tags(f, bands).iloc[::-1]
    #         df.insert(0, 'GeoPotential', data)
    #         df.insert(1, 'GeoPotential_Height', scaled_data)

    #         return df, md

    # if data is None:
    #     raise AncillaryError("No ECWMF Geo-Potential profile data")


def ecwmf_relative_humidity(input_path, lonlat, time):
    """Retrieve a pixel value from the ECWMF Relative Humidity collection
    across 37 height pressure levels, for a given longitude,
    latitude and time.

    Reverses the order of elements
    (1000 -> 1 mb, rather than 1 -> 1000 mb) before returning.
    """
    # TODO; have swfo convert the files to HDF5
    raise AncillaryError("No ECWMF Relative Humidity profile data")
    # product = DatasetName.RELATIVE_HUMIDITY.value.lower()
    # search = pjoin(input_path, DatasetName.ECMWF_PATH_FMT.value)
    # files = glob.glob(search.format(product=product, year=time.year))
    # data = None
    # required_ymd = datetime.datetime(time.year, time.month, time.day)
    # for f in files:
    #     url = urlparse(f, scheme='file').geturl()
    #     ymd = splitext(basename(f))[0].split('_')[1]
    #     ancillary_ymd = datetime.datetime.strptime(ymd, '%Y-%m-%d')
    #     if ancillary_ymd == required_ymd:
    #         bands = list(range(1, 38))
    #         data = get_pixel(f, lonlat, bands)[::-1]

    #         metadata = {'data_source': 'ECWMF Relative Humidity',
    #                     'url': url,
    #                     'query_date': time}

    #         # file level metadata
    #         md = extract_ancillary_metadata(f)
    #         for key in md:
    #             metadata[key] = md[key]

    #         # internal file metadata (and reverse the ordering)
    #         df = read_metadata_tags(f, bands).iloc[::-1]
    #         df.insert(0, 'Relative_Humidity', data)

    #         return df, metadata

    # if data is None:
    #     raise AncillaryError("No ECWMF Relative Humidity profile data")
