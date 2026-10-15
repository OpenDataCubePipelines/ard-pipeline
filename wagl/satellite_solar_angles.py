"""Satellite and Solar angle claculations over a 2D grid."""

import math

import ephem
import numpy as np
import shapely
import shapely.ops
from osgeo import osr
from pyproj import Transformer as ProjTransformer

from wagl.__sat_sol_angles import satellite_angle as satellite_angle_prim
from wagl.__sat_sol_angles import solar_angle as solar_angle_prim
from wagl.__satellite_model import set_satmod
from wagl.__track_time_info import set_times
from wagl.constants import DatasetName, GroupName
from wagl.hdf5 import (
    H5CompressionFilter,
    attach_attributes,
    attach_image_attributes,
    attach_table_attributes,
    write_scalar,
)
from wagl.tle import load_tle

CRS = "EPSG:4326"


def convert_pixel_index_to_projected_coord(geobox, pixel_indices, centre=True):
    shift = 0.5 if centre else 0.0
    return np.array(
        [(geobox.transform * point) for point in (pixel_indices + shift).tolist()]
    )


def convert_projected_coord_to_pixel_index(geobox, projected_coords, centre=True):
    shift = 0.5 if centre else 0.0
    pixels = np.array(
        [((~geobox.transform) * point) for point in projected_coords.tolist()]
    )
    return np.round(pixels - shift).astype(np.int32)


def convert_projected_coord_to_geographic(geobox, projected_coords):
    transformer = ProjTransformer.from_crs(
        geobox.crs.ExportToWkt(), CRS, always_xy=True
    )
    lon, lat = transformer.transform(projected_coords[:, 0], projected_coords[:, 1])
    return np.hstack([lon[:, np.newaxis], lat[:, np.newaxis]])


def convert_geographic_to_projected_coord(geobox, lonlat):
    transformer = ProjTransformer.from_crs(
        CRS, geobox.crs.ExportToWkt(), always_xy=True
    )
    map_x, map_y = transformer.transform(lonlat[:, 0], lonlat[:, 1])
    return np.hstack([map_x[:, np.newaxis], map_y[:, np.newaxis]])


# not used, kept here as a reminder of the possible optimization
# that the coordinator points need to be outside a certain
# range of satellite view angles
def first_and_last(array):
    """Utility to find indices of first and last true value in an array.

    Example:
    -------
                            0 1 2 3 4 5 6 7 8 9
                                |       |
        >>> first_and_last([0,0,1,1,0,1,1,0,0,0])
        (2, 6)
        >>> first_and_last([0,0,0]) # if no element of truth:
        (-1, -1)
        >>> first_and_last([1])
        (0, 0)
    """
    (i,) = np.nonzero(array)  # assume array only has one dimension to unpack
    return (i[0], i[-1]) if i.shape[0] else (-1, -1)


# not used, kept here as a reminder of the possible optimization
# that the coordinator points need to be outside a certain
# range of satellite view angles
def swathe_edges(threshold, array):
    """Find left and right edges of swathe.

    Takes raster array, compares values against threshold, and returns a pair
    of vectors which represent the indices of the first and last pixel in
    each row of the array.
    """
    start = np.empty(array.shape[0], dtype="int")
    end = np.empty(array.shape[0], dtype="int")
    for i, row in enumerate(array):
        start[i], end[i] = first_and_last(row <= threshold)
    return start, end


def create_coordinator(locations, geobox):
    """
    :param locations:
        A 2D `NumPy` array of shape (N, 3) containing
        pixel coordinates of the sample locations
        and the geometry component label.

    :return:
    A `NumPy` dataset of the following datatype:

    * np.dtype([('row_index', 'int64'),
    ('col_index', 'int64'),
    ('latitude', 'float64'),
    ('longitude', 'float64'),
    ('map_y', 'int64'),
    ('map_x', 'int64'),
    ('geom', 'int32')])
    """
    # custom datatype for coordinator
    coordinator_dtype = np.dtype(
        [
            ("row_index", "int64"),
            ("col_index", "int64"),
            ("latitude", "float64"),
            ("longitude", "float64"),
            ("map_y", "int64"),
            ("map_x", "int64"),
            ("geom", "int32"),
        ]
    )
    coordinator = np.empty(locations.shape[0], dtype=coordinator_dtype)
    coordinator["col_index"] = locations[:, 0]
    coordinator["row_index"] = locations[:, 1]
    coordinator["geom"] = locations[:, 2]

    projected = convert_pixel_index_to_projected_coord(geobox, locations[:, :2])
    coordinator["map_x"] = projected[:, 0]
    coordinator["map_y"] = projected[:, 1]

    lonlat = convert_projected_coord_to_geographic(geobox, projected)
    coordinator["longitude"] = lonlat[:, 0]
    coordinator["latitude"] = lonlat[:, 1]

    return coordinator


def create_track_linestring(geobox, track_dset):
    track_lon = np.rad2deg(track_dset["lam"])
    track_lat = np.rad2deg(track_dset["phi_p"])

    lonlat = np.hstack([track_lon[:, np.newaxis], track_lat[:, np.newaxis]])
    projected = convert_geographic_to_projected_coord(geobox, lonlat)
    points = convert_projected_coord_to_pixel_index(geobox, projected)

    return shapely.LineString(shapely.points(points[:, 0], points[:, 1]))


def create_regular_grid_points(geobox, vertices):
    rows, cols = geobox.shape

    y = np.linspace(0, rows - 1, vertices[0]).astype("int32")
    x = np.linspace(0, cols - 1, vertices[1]).astype("int32")

    xx, yy = np.meshgrid(x, y, indexing="xy")
    col_index = xx.ravel()
    row_index = yy.ravel()

    return shapely.multipoints(shapely.points(col_index, row_index))


def split_grid_by_track(grid_points, track_string, vertices):
    # TODO or should vertices be (y_vertices, x_vertices, track_points)?
    assert vertices[0] == vertices[1]

    grid_hull = grid_points.convex_hull

    split_hull = shapely.ops.split(grid_hull, track_string)
    assert split_hull.geom_type == "GeometryCollection"

    parts = list(split_hull.geoms)
    assert len(parts) != 0 and len(parts) <= 2

    if len(parts) == 1:
        return {"grid": grid_points}

    track_clipped = shapely.multipoints(
        track_string.intersection(grid_hull).interpolate(
            np.linspace(0, 1, vertices[0]), normalized=True
        )
    )

    part1, part2 = parts

    points1 = shapely.multipoints([p for p in grid_points.geoms if part1.covers(p)])
    points2 = shapely.multipoints([p for p in grid_points.geoms if part2.covers(p)])

    assert len(grid_points.geoms) == len(points1.geoms) + len(points2.geoms)

    return {
        "grid": grid_points,
        "track": track_clipped,
        "part1": points1,
        "part2": points2,
    }


def coordinator_geometries(geobox, track, vertices):
    track_linestring = create_track_linestring(geobox, track)
    grid_points = create_regular_grid_points(geobox, vertices)
    return split_grid_by_track(grid_points, track_linestring, vertices)


def create_vertices(acquisition, track, vertices=(3, 3)):
    """Defines the point locations, and the number of points, across a
    spatial grid.

    :param acquisition:
        An instance of an `Acquisition` object.

    :param track:
        A dict containing satellite track coordinates.

    :param vertices:
        An integer 2-tuple indicating the number of rows and columns
        of sample-locations ("coordinator") to produce.
        The vertex columns should be an odd number.
        Default is (3, 3).

    :return:
        A `NumPy` dataset of the following datatype:

        * np.dtype([('row_index', 'int64'),
                    ('col_index', 'int64'),
                    ('latitude', 'float64'),
                    ('longitude', 'float64'),
                    ('map_y', 'int64'),
                    ('map_x', 'int64')])
    """
    geobox = acquisition.gridded_geo_box()
    geoms = coordinator_geometries(geobox, track, vertices)

    def assign_component_label(points, label):
        coords = shapely.get_coordinates(points)
        tag = np.full((coords.shape[0], 1), label)
        return np.hstack([coords, tag])

    if "track" not in geoms:
        # 0: no track intersection, all points from grid
        points = assign_component_label(geoms["grid"], 0)
    else:
        # 1: track points
        # 2: component 1
        # 3: component 2
        points = np.vstack(
            [
                assign_component_label(geoms["track"], 1),
                assign_component_label(geoms["part1"], 2),
                assign_component_label(geoms["part2"], 3),
            ]
        )

    return create_coordinator(points, geobox)


def calculate_julian_century(datetime):
    """Given a datetime object return the julian century from the 2000 epoch.

    :param datetime:
        A datetime object containing the date to be converted to a
        Julian centuries since 2000/01/01 12:00.

    :return:
        A floating point value representing the Julian centuries since
        the 2000/01/01 12:00 epoch.
    """
    # Convert the scene timestamp to a julian date
    d = ephem.date(datetime)
    jdate = ephem.julian_date(d)

    # Get the J2000 epoch
    epoch = ephem.date((2000, 1, 1, 12.00))
    j2_epoch = ephem.julian_date(epoch)

    # Note:
    # This differs from online sources such as
    # http://www.pietro.org/astro_util_staticdemo/FDetailDateConversions.htm
    # http://en.wikipedia.org/wiki/Equinox_(celestial_coordinates)
    # which use:
    # 2000 + (jdate - epoch) / 365.25
    century = (jdate - j2_epoch) / 36525

    return century


def setup_spheroid(proj_wkt):
    """Given a WKT projection string, determine the spheroid parameters
    that will be used in calculating the angle grids.

    :param proj_wkt:
        A string containing valid WKT projection information.

    :return:
        A floating point np array of 4 elements containing
        spheroidal parameters.

            * Index 0 contains the spheroid Major Axis.
            * Index 1 contains the spheroid Inverse Flattening.
            * Index 2 contains the spheroid Squared Eccentricity.
            * Index 3 contains the Earth rotational angular velocity in
              radians/second.

        Also a np dataset of the following datatype:

            * dtype = [('semi_major_axis', 'float64'),
                       ('inverse_flattening', 'float64'),
                       ('eccentricity_squared', 'float64'),
                       ('earth_rotational_angular_velocity', 'float64')]
    """
    dtype = np.dtype(
        [
            ("semi_major_axis", "float64"),
            ("inverse_flattening", "float64"),
            ("eccentricity_squared", "float64"),
            ("earth_rotational_angular_velocity", "float64"),
        ]
    )
    dset = np.zeros(1, dtype=dtype)

    # Define the spatial reference
    sr = osr.SpatialReference()
    sr.ImportFromWkt(proj_wkt)

    # Spheroid major axis
    dset["semi_major_axis"] = sr.GetSemiMajor()

    # Inverse flattening
    dset["inverse_flattening"] = sr.GetInvFlattening()

    # Eccentricity squared
    dset["eccentricity_squared"] = 1.0 - (1.0 - 1.0 / dset["inverse_flattening"]) ** 2

    # Earth rotational angular velocity rad/sec
    # Other sources such as:
    # http://www.oosa.unvienna.org/pdf/icg/2012/template/WGS_84.pdf
    # state 0.000072921150 as the mean value
    dset["earth_rotational_angular_velocity"] = 0.000072722052

    return np.array(dset.tolist()).squeeze(), dset


def setup_orbital_elements(acquisition, tle_path):
    """Given an ephemeral object and a datetime object, calculate the
    satellite orbital paramaters used for calculating the angle grids.

    :param acquisition:
        An `Acquisition` object.

    :param tle_path:
        A `str` to the directory containing the Two Line Element data.

    :return:
        A floating point np array of 3 elements containing the
        satellite ephemeral bodies orbital paramaters.

            * Index 0 contains the obrital inclination in degrees.
            * Index 1 contains the semi major raidus in metres.
            * Index 2 contains the angular velocity in radians/sec^1.

        Also a np dataset of the following datatype:

            * dtype = [('orbital_inclination', 'float64'),
                       ('semi_major_radius', 'float64'),
                       ('angular_velocity', 'float64')]
    """
    dtype = np.dtype(
        [
            ("orbital_inclination", "float64"),
            ("semi_major_radius", "float64"),
            ("angular_velocity", "float64"),
        ]
    )
    dset = np.zeros(1, dtype=dtype)

    ephemeral = load_tle(acquisition, tle_path)

    # If we have None, then no suitable TLE was found, so use values gathered
    # by the acquisition object
    if ephemeral is None:
        # orbital inclination (degrees)
        dset["orbital_inclination"] = math.degrees(acquisition.inclination)
        # semi_major radius (m)
        dset["semi_major_radius"] = acquisition.semi_major_axis
        # angular velocity (rad sec-1)
        dset["angular_velocity"] = acquisition.omega
    else:
        ephemeral.compute(acquisition.acquisition_datetime)
        pi = np.pi
        n = ephemeral._n  # number or orbits per day
        s = 24 * 60 * 60  # Seconds in a day
        mu = 398600441800000.0  # Earth Gravitational parameter m^3s^-2

        # orbital inclination (degrees)
        dset["orbital_inclination"] = np.rad2deg(ephemeral._inc)

        # semi_major radius (m)
        # http://smallsats.org/2012/12/06/two-line-element-set-tle/
        dset["semi_major_radius"] = (mu / (2 * pi * n / s) ** 2) ** (1 / 3)

        # angular velocity (rad sec-1)
        dset["angular_velocity"] = (2 * pi * n) / s

    return np.array(dset.tolist()).squeeze(), dset


def setup_smodel(centre_lon, centre_lat, spheroid, orbital_elements):
    """Setup the satellite model.
    A wrapper routine for the `set_satmod` Fortran module built via
    ``F2Py``.

    :param centre_lon:
        The longitude of the scene centre.

    :param centre_lat:
        The lattitude of the scene centre.

    :param spheroid:
        A 4 element floating point array containing the Earth
        spheroidal paramaters.

            * Index 0 contains the spheroid Major Axis.
            * Index 1 contains the spheroid Inverse Flattening.
            * Index 2 contains the spheroid Squared Eccentricity.
            * Index 3 contains the Earth rotational angular velocity in
              radians/second.

    :param orbital_elements:
        A 3 element floating point array containing the satellite
        orbital elements.

            * Index 0 contains the obrital inclination in degrees.
            * Index 1 contains the semi major raidus in metres.
            * Index 2 contains the angular velocity in radians/sec^1.

    :return:
        A floating point np array of 30 elements containing the
        satellite model paramaters.

            * Index 0 contains phi0.
            * Index 1 contains phi0_p.
            * Index 2 contains rho0.
            * Index 3 contains t0.
            * Index 4 contains lam0.
            * Index 5 contains gamm0.
            * Index 6 contains beta0.
            * Index 7 contains rotn0.
            * Index 8 contains hxy0.
            * Index 9 contains N0.
            * Index 10 contains H0.
            * Index 11 contains th_ratio0.

        Also np dataset of the following datatype:

            * dtype = [('phi0', 'f8'), ('phi0_p', 'f8'), ('rho0', 'f8'),
                       ('t0', 'f8'), ('lam0', 'f8'), ('gamm0', 'f8'),
                       ('beta0', 'f8'), ('rotn0', 'f8'), ('hxy0', 'f8'),
                       ('N0', 'f8'), ('H0', 'f8'), ('th_ratio0', 'f8')]
    """
    smodel, _ = set_satmod(centre_lon, centre_lat, spheroid, orbital_elements)

    columns = [
        "phi0",
        "phi0_p",
        "rho0",
        "t0",
        "lam0",
        "gamm0",
        "beta0",
        "rotn0",
        "hxy0",
        "N0",
        "H0",
        "th_ratio0",
    ]
    dtype = np.dtype([(col, "float64") for col in columns])
    smodel_dset = np.array([tuple(smodel)], dtype=dtype)

    return smodel, smodel_dset


def setup_times(ymin, ymax, spheroid, orbital_elements, smodel, ntpoints=30):
    """Setup the satellite track times.
    A wrapper routine for the ``set_times`` Fortran module built via
    ``F2Py``.

    :param ymin:
        The minimum lattitude in the array extent.

    :param ymax:
        The maximum lattitude in the array extent.

    :param spheroid:
        A 4 element floating point array containing the Earth
        spheroidal paramaters.

            * Index 0 contains the spheroid Major Axis.
            * Index 1 contains the spheroid Inverse Flattening.
            * Index 2 contains the spheroid Squared Eccentricity.
            * Index 3 contains the Earth rotational angular velocity in
              radians/second.

    :param orbital_elements:
        A 3 element floating point array containing the satellite
        orbital elements.

            * Index 0 contains the obrital inclination in degrees.
            * Index 1 contains the semi major raidus in metres.
            * Index 2 contains the angular velocity in radians/sec^1.

    :param smodel:
        A floating point np array of 30 elements containing the
        satellite model paramaters:

            * Index 0 contains phi0.
            * Index 1 contains phi0_p.
            * Index 2 contains rho0.
            * Index 3 contains t0.
            * Index 4 contains lam0.
            * Index 5 contains gamm0.
            * Index 6 contains beta0.
            * Index 7 contains rotn0.
            * Index 8 contains hxy0.
            * Index 9 contains N0.
            * Index 10 contains H0.
            * Index 11 contains th_ratio0.

    :param ntpoints:
        The number of time sample points to be calculated along the
        satellite track. Default is 30.

    :return:
        A floating point np array of [ntpoints,8] containing the
        satellite track times and other information.

            * Index 0 t.
            * Index 1 rho.
            * Index 2 phi_p.
            * Index 3 lam.
            * Index 4 beta.
            * Index 5 hxy.
            * Index 6 mj.
            * Index 7 skew.

        Also a np dataset of the datatype:

            * dtype = [('t', 'f8'), ('rho', 'f8'), ('phi_p', 'f8'),
                       ('lam', 'f8'), ('beta', 'f8'), ('hxy', 'f8'),
                       ('mj', 'f8'), ('skew', 'f8')]
    """
    track = set_times(ymin, ymax, ntpoints, spheroid, orbital_elements, smodel)

    columns = ["t", "rho", "phi_p", "lam", "beta", "hxy", "mj", "skew"]
    dtype = np.dtype([(col, "float64") for col in columns])
    track_dset = np.zeros(ntpoints, dtype=dtype)
    track_dset["t"] = track[:, 0]
    track_dset["rho"] = track[:, 1]
    track_dset["phi_p"] = track[:, 2]
    track_dset["lam"] = track[:, 3]
    track_dset["beta"] = track[:, 4]
    track_dset["hxy"] = track[:, 5]
    track_dset["mj"] = track[:, 6]
    track_dset["skew"] = track[:, 7]

    return track, track_dset


def _store_parameter_settings(
    fid, spheriod, orbital_elements, satellite_model, satellite_track, params
):
    """An internal function for storing the parameter settings for the
    calculate_angles workflow.
    """
    group = fid.create_group("PARAMETERS")

    # generic parameters
    dname = DatasetName.GENERIC.value
    write_scalar("GENERIC PARAMETERS", dname, group, params)

    # sheroid
    desc = "The spheroid used in the satellite and solar angles calculation."
    attrs = {"description": desc}
    dname = DatasetName.SPHEROID.value
    sph_dset = group.create_dataset(dname, data=spheriod)
    attach_table_attributes(sph_dset, title="Spheroid", attrs=attrs)

    # orbital elements
    desc = (
        "The satellite orbital parameters used in the satellite and "
        "solar angles calculation."
    )
    attrs = {"description": desc}
    dname = DatasetName.ORBITAL_ELEMENTS.value
    orb_dset = group.create_dataset(dname, data=orbital_elements)
    attach_table_attributes(orb_dset, title="Orbital Elements", attrs=attrs)

    # satellite model
    desc = "The satellite model used in the satellite and solar angles " "calculation."
    attrs = {"description": desc}
    dname = DatasetName.SATELLITE_MODEL.value
    sat_dset = group.create_dataset(dname, data=satellite_model)
    attach_table_attributes(sat_dset, title="Satellite Model", attrs=attrs)

    # satellite track
    desc = (
        "The satellite track information used in the satellite and solar "
        "angles calculation."
    )
    attrs = {"description": desc}
    dname = DatasetName.SATELLITE_TRACK.value
    track_dset = group.create_dataset(dname, data=satellite_track)
    attach_table_attributes(track_dset, title="Satellite Track", attrs=attrs)


def solar_angle(nrow, ncol, alat, alon, hours, century, asol, soazi):
    solar_angle_prim(nrow, ncol, alat, alon, hours, century, asol, soazi)


def satellite_angle(
    nrow,
    ncol,
    row_offset,
    col_offset,
    alat,
    alon,
    spheroid,
    orb_elements,
    ntpoints,
    smodel,
    track,
    view,
    azi,
    tim,
):
    stat = satellite_angle_prim(
        nrow,
        ncol,
        row_offset,
        col_offset,
        alat,
        alon,
        spheroid,
        orb_elements,
        ntpoints,
        smodel,
        track,
        view,
        azi,
        tim,
    )

    if np.any(stat != 0):
        msg = "Error in calculating angles, no interval found in track!"
        raise RuntimeError(msg)


def calculate_angles(
    acquisition,
    lon_lat_group,
    out_group=None,
    compression=H5CompressionFilter.LZF,
    filter_opts=None,
    tle_path=None,
    trackpoints=30,
):
    """Calculate the satellite view, satellite azimuth, solar zenith,
    solar azimuth, and relative aziumth angle grids, as well as the
    time grid.

    :param acquisition:
        An instance of an `Acquisition` object.

    :param lon_lat_group::
        The root HDF5 `Group` that contains the longitude and
        latitude datasets.
        The dataset pathnames are given by:

        * DatasetName.LON
        * DatasetName.LAT

    :param out_group:
        A writeable HDF5 `Group` object.

        The dataset names will be as follows:

        * DatasetName.SATELLITE_VIEW
        * DatasetName.SATELLITE_AZIMUTH
        * DatasetName.SOLAR_ZENITH
        * DatasetName.SOLAR_AZIMUTH
        * DatasetName.RELATIVE_AZIMUTH
        * DatasetName.TIME
        * DatasetName.CENTRELINE
        * DatasetName.BOXLINE
        * DatasetName.SPHEROID
        * DatasetName.ORBITAL_ELEMENTS
        * DatasetName.SATELLITE_MODEL
        * DatasetName.SATELLITE_TRACK

    :param trackpoints:
        Number of trackpoints to use when calculating solar angles
        Default is 30 to detect track intersecting acquisition.

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

    :param tle_path:
        A `str` to the directory containing the Two Line Element data.

    :param trackpoints:
        TODO.

    :return:
        An opened `h5py.File` object, that is either in-memory using the
        `core` driver, or on disk.
    """
    century = calculate_julian_century(acquisition.acquisition_datetime)
    geobox = acquisition.gridded_geo_box()

    # longitude and latitude datasets
    longitude = lon_lat_group[DatasetName.LON.value]
    latitude = lon_lat_group[DatasetName.LAT.value]

    # Min and Max lat extents
    # This method should handle northern and southern hemispheres
    # TODO: Put in a conditional over the 1 degree buffer
    min_lat = (
        min(
            min(geobox.ul_lonlat[1], geobox.ur_lonlat[1]),
            min(geobox.ll_lonlat[1], geobox.lr_lonlat[1]),
        )
        - 1
    )
    max_lat = (
        max(
            max(geobox.ul_lonlat[1], geobox.ur_lonlat[1]),
            max(geobox.ll_lonlat[1], geobox.lr_lonlat[1]),
        )
        + 1
    )

    # Get the lat/lon of the scene centre
    # check if we have a file with GPS satellite track points
    # which can be used for cases of image granules/tiles, eg Sentinel-2A
    if acquisition.gps_file:
        points = acquisition.read_gps_file()
        subs = points[(points.latitude >= min_lat) & (points.latitude <= max_lat)]
        idx = subs.shape[0] // 2 - 1
        centre_xy = (subs.iloc[idx].longitude, subs.iloc[idx].latitude)
    else:
        centre_xy = geobox.centre_lonlat

    # Get the earth spheroidal paramaters
    spheroid = setup_spheroid(geobox.crs.ExportToWkt())

    # Get the satellite orbital elements
    orbital_elements = setup_orbital_elements(acquisition, tle_path)

    # Get the satellite model paramaters

    smodel = setup_smodel(centre_xy[0], centre_xy[1], spheroid[0], orbital_elements[0])

    # Get the times and satellite track information
    track = setup_times(
        min_lat,
        max_lat,
        spheroid[0],
        orbital_elements[0],
        smodel[0],
        trackpoints,
    )

    assert out_group is not None
    fid = out_group

    if GroupName.SAT_SOL_GROUP.value not in fid:
        fid.create_group(GroupName.SAT_SOL_GROUP.value)

    grp = fid[GroupName.SAT_SOL_GROUP.value]

    # store the parameter settings used with the satellite and solar angles
    # function
    params = {
        "dimensions": (acquisition.lines, acquisition.samples),
        "lines": acquisition.lines,
        "samples": acquisition.samples,
        "century": century,
        "decimal_hour": acquisition.decimal_hour(),
        "acquisition_datetime": acquisition.acquisition_datetime,
        "centre_longitude_latitude": centre_xy,
        "minimum_latiude": min_lat,
        "maximum_latiude": max_lat,
        "latitude_buffer": 1.0,
        "max_view_angle": acquisition.maximum_view_angle,
    }
    _store_parameter_settings(
        grp, spheroid[1], orbital_elements[1], smodel[1], track[1], params
    )

    out_dtype = "float32"
    no_data = np.nan
    kwargs = compression.settings(filter_opts, chunks=acquisition.tile_size)
    kwargs["shape"] = (acquisition.lines, acquisition.samples)
    kwargs["fillvalue"] = no_data
    kwargs["dtype"] = out_dtype

    sat_v_ds = grp.create_dataset(DatasetName.SATELLITE_VIEW.value, **kwargs)
    sat_az_ds = grp.create_dataset(DatasetName.SATELLITE_AZIMUTH.value, **kwargs)
    sol_z_ds = grp.create_dataset(DatasetName.SOLAR_ZENITH.value, **kwargs)
    sol_az_ds = grp.create_dataset(DatasetName.SOLAR_AZIMUTH.value, **kwargs)
    rel_az_ds = grp.create_dataset(DatasetName.RELATIVE_AZIMUTH.value, **kwargs)
    time_ds = grp.create_dataset(DatasetName.TIME.value, **kwargs)

    # base attributes for image datasets
    attrs = {
        "crs_wkt": geobox.crs.ExportToWkt(),
        "geotransform": geobox.transform.to_gdal(),
        "no_data_value": no_data,
    }
    attach_image_attributes(sat_v_ds, attrs)
    attach_image_attributes(sat_az_ds, attrs)
    attach_image_attributes(sol_z_ds, attrs)
    attach_image_attributes(sol_az_ds, attrs)
    attach_image_attributes(rel_az_ds, attrs)
    attach_image_attributes(time_ds, attrs)

    attrs = {
        "description": "Contains the satellite viewing angle in degrees.",
        "units": "degrees",
        "alias": "satellite-view",
    }
    attach_attributes(sat_v_ds, attrs)

    attrs = {
        "description": "Contains the satellite azimuth angle in degrees.",
        "units": "degrees",
        "alias": "satellite-azimuth",
    }
    attach_attributes(sat_az_ds, attrs)

    attrs = {
        "description": "Contains the solar zenith angle in degrees.",
        "units": "degrees",
        "alias": "solar-zenith",
    }
    attach_attributes(sol_z_ds, attrs)

    attrs = {
        "description": "Contains the solar azimuth angle in degrees.",
        "units": "degrees",
        "alias": "solar-azimuth",
    }
    attach_attributes(sol_az_ds, attrs)

    attrs = {
        "description": "Contains the relative azimuth angle in degrees.",
        "units": "degrees",
        "alias": "relative-azimuth",
    }
    attach_attributes(rel_az_ds, attrs)

    attrs = {
        "description": "Contains the time from apogee in seconds.",
        "units": "seconds",
        "alias": "timedelta",
    }
    attach_attributes(time_ds, attrs)

    for tile in acquisition.tiles(xtol=1, ytol=1):
        idx = (slice(tile[0][0], tile[0][1]), slice(tile[1][0], tile[1][1]))

        # read the lon and lat tile
        lon_data = np.asfortranarray(longitude[idx])
        lat_data = np.asfortranarray(latitude[idx])

        # may not be processing full row wise (all columns)
        dims = lon_data.shape
        row_offset = idx[0].start
        col_offset = idx[1].start

        view = np.asfortranarray(np.full(dims, no_data, dtype=out_dtype))
        azi = np.asfortranarray(np.full(dims, no_data, dtype=out_dtype))
        asol = np.asfortranarray(np.full(dims, no_data, dtype=out_dtype))
        soazi = np.asfortranarray(np.full(dims, no_data, dtype=out_dtype))
        time = np.asfortranarray(np.full(dims, no_data, dtype=out_dtype))

        solar_angle(
            dims[0],
            dims[1],
            lat_data,
            lon_data,
            acquisition.decimal_hour(),
            century,
            asol,
            soazi,
        )

        satellite_angle(
            dims[0],
            dims[1],
            row_offset,
            col_offset,
            lat_data,
            lon_data,
            spheroid[0],
            orbital_elements[0],
            trackpoints,
            smodel[0],
            track[0],
            view,
            azi,
            time,
        )

        # output to disk
        sat_v_ds[idx] = view
        sat_az_ds[idx] = azi
        sol_z_ds[idx] = asol
        sol_az_ds[idx] = soazi
        rel_az_ds[idx] = azi - soazi
        time_ds[idx] = time
