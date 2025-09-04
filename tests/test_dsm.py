# Partial tests for the DSM/Digital Surface Model module
import pytest

from wagl import dsm

# Extents fixtures: these provide lat/long extents of several global Landsat
# scenes. These are intended to cover northern/southern & east/west hemispheres.
# Antarctic areas are also covered to include an area of polar extremes.


@pytest.fixture
def l9_antarctic_volcano_extents():
    # Extents from LC09_L1GT_007114_20241215_20241215_02_T2.tar, converted from
    # original to WGS84 CRS with `geobox.project_extents(wgs84_crs)`

    # order is min_x, min_y, max_x, max_y
    return -119.9633131, -77.2715533, -108.62444196, -74.51953282


@pytest.fixture
def l5_wagga_extents():
    # From LT05_L1TP_092084_20080925_20161029_01_T1.tar
    return 145.434996, -35.581614, 148.077655, -33.653483


@pytest.fixture
def l9_buenos_aires_extents():
    # Extents converted from LC09_L1TP_225084_20241222_20241222_02_T1.tar

    # order is min_x, min_y, max_x, max_y
    return -60.05997761, -35.66956971, -57.50299335, -33.54584702


@pytest.fixture
def l9_italy_extents():
    # from LC09_L1TP_188033_20231030_20231030_02_T1.tar
    return 14.491808, 37.828106, 17.192353, 39.956304


@pytest.fixture
def l9_arizona_extents():
    # from LC09_L1TP_037036_20241030_20241030_02_T1.tar
    return -113.457772, 33.523343, -110.889247, 35.675949


@pytest.fixture
def l9_fiji_extents():
    # From Fiji scene LC90730712024364LGN00:
    # Corner Upper Left Latitude 	-14.84924
    # Corner Upper Left Longitude 	179.68416  # west of antimeridian
    #
    # Corner Upper Right Latitude 	-14.87003
    # Corner Upper Right Longitude 	-178.20377
    #
    # Corner Lower Left Latitude 	-16.93133
    # Corner Lower Left Longitude 	179.64985  # east of antimeridian
    #
    # Corner Lower Right Latitude 	-16.95520
    # Corner Lower Right Longitude 	-178.21625
    #
    # use same min/max lat/long order
    return -178.20377, -16.95520, 179.64985, -14.84924


# Section: helper functions


def get_unique_lat_longs_from_coord_pairs(tuple_sequence):
    """
    Split (lat, long) tuple sequence into separate latitude/longitude sequences.
    """
    latitudes, longitudes = zip(*tuple_sequence)
    return set(latitudes), set(longitudes)


# Section: basic extents testing


def test_copdem_lat_long_extents_southern_hemisphere_polar(
    l9_antarctic_volcano_extents,
):
    # Ensure CopDEM tile search provides correct coverage for Landsat scene extents
    gen = dsm.copernicus_tiles_latlon_covering_extents(l9_antarctic_volcano_extents)
    latitudes, longitudes = get_unique_lat_longs_from_coord_pairs(gen)

    assert latitudes == {-78, -77, -76, -75}
    assert longitudes == set(range(-120, -108))


def test_copdem_lat_long_extents_southern_and_eastern_hemispheres(l5_wagga_extents):
    gen = dsm.copernicus_tiles_latlon_covering_extents(l5_wagga_extents)
    latitudes, longitudes = get_unique_lat_longs_from_coord_pairs(gen)

    # these lat/long extents confirmed by loading CopDEM tiles in QGIS
    assert latitudes == {-36, -35, -34}
    assert longitudes == {145, 146, 147, 148}


def test_copdem_lat_long_extents_southern_and_western_hemispheres(
    l9_buenos_aires_extents,
):
    gen = dsm.copernicus_tiles_latlon_covering_extents(l9_buenos_aires_extents)
    latitudes, longitudes = get_unique_lat_longs_from_coord_pairs(gen)

    assert latitudes == {-34, -35, -36}
    assert longitudes == {-61, -60, -59, -58}


def test_copdem_lat_long_extents_northern_and_eastern_hemispheres(l9_italy_extents):
    gen = dsm.copernicus_tiles_latlon_covering_extents(l9_italy_extents)
    latitudes, longitudes = get_unique_lat_longs_from_coord_pairs(gen)

    assert latitudes == {37, 38, 39}
    assert longitudes == {14, 15, 16, 17}


def test_copdem_lat_long_extents_northern_and_western_hemispheres(l9_arizona_extents):
    gen = dsm.copernicus_tiles_latlon_covering_extents(l9_arizona_extents)
    latitudes, longitudes = get_unique_lat_longs_from_coord_pairs(gen)

    assert latitudes == {33, 34, 35}
    assert longitudes == {-114, -113, -112, -111}


# Section: more detailed lat/long tests
# confirm the exact (lat, long) extents pairs for covering tiles


def test_eastern_hemisphere_lat_long_pairs(l5_wagga_extents):
    gen = dsm.copernicus_tiles_latlon_covering_extents(l5_wagga_extents)

    for lat in (-36, -35, -34):
        for long in (145, 146, 147, 148):
            assert (lat, long) == gen.__next__()


def test_western_hemisphere_lat_long_pairs(l9_antarctic_volcano_extents):
    gen = dsm.copernicus_tiles_latlon_covering_extents(l9_antarctic_volcano_extents)

    for lat in (-78, -77, -76, -75):
        for long in range(-120, -108):
            assert (lat, long) == gen.__next__()


# Section: equatorial crossing tests - ensure extents work over the equator


def test_eastern_hemisphere_extents_crossing_equator():
    # copy & shift Wagga scene extent latitudes north to mimic eastern hemisphere
    # scene over the equator
    min_long = 145.434996
    min_lat = -35.581614 + 34.0
    max_long = 148.077655
    max_lat = -33.653483 + 35.0  # artificially extend by an extra degree

    # ensure latitude range crosses equator
    assert min_lat < 0
    assert max_lat > 0

    extents = (min_long, min_lat, max_long, max_lat)
    gen = dsm.copernicus_tiles_latlon_covering_extents(extents)
    latitudes, longitudes = get_unique_lat_longs_from_coord_pairs(gen)

    assert latitudes == {-2, -1, 0, 1}
    assert longitudes == {145, 146, 147, 148}


def test_western_hemisphere_extents_crossing_equator():
    # copy & shift Buenos Aires scene extent latitudes north to mimic western
    # hemisphere scene over the equator
    min_long = -60.05997761
    min_lat = -35.66956971 + 34.0
    max_long = -57.50299335
    max_lat = -33.54584702 + 35.0

    # ensure latitude range crosses equator
    assert min_lat < 0
    assert max_lat > 0

    extents = (min_long, min_lat, max_long, max_lat)
    gen = dsm.copernicus_tiles_latlon_covering_extents(extents)
    latitudes, longitudes = get_unique_lat_longs_from_coord_pairs(gen)

    assert latitudes == {-2, -1, 0, 1}
    assert longitudes == {-61, -60, -59, -58}


# Section: antimeridian +/- 180 detection tests


def test_intersects_antimeridian(l9_fiji_extents):
    assert dsm.intersects_antimeridian(l9_fiji_extents)
    assert not dsm.disjoint_antimeridian(l9_fiji_extents)


def test_intersects_prime_meridian():
    # ensure false positives are avoided, e.g. at the prime meridian (0 degrees)
    extents = (-1.0, 10.0, 1.0, 12.0)
    assert not dsm.intersects_antimeridian(extents)
    assert dsm.disjoint_antimeridian(extents)


def test_disjoint_antimeridian(l5_wagga_extents, l9_arizona_extents, l9_italy_extents):
    # avoid false positives
    assert dsm.disjoint_antimeridian(l5_wagga_extents)
    assert dsm.disjoint_antimeridian(l9_arizona_extents)
    assert dsm.disjoint_antimeridian(l9_italy_extents)


# Section: test coordinate ranges at the antimeridian


def test_fiji_extents_crossing_antimeridian(l9_fiji_extents):
    gen = dsm.copernicus_tiles_latlon_covering_extents(l9_fiji_extents)
    latitudes, longitudes = get_unique_lat_longs_from_coord_pairs(gen)

    assert latitudes == {-17, -16, -15}
    assert longitudes == {-180, -179, 179}


def test_southern_hemisphere_extents_crossing_antimeridian():
    # copy & shift Wagga scene extent longitudes to cross antimeridian
    min_long = -177.922345  # -180 + width of ~2.07 degrees
    min_lat = -35.581614
    max_long = 179.434996  # from 145.434996 + 34.0
    max_lat = -33.653483

    # partially ensure longitude range crosses antimeridian
    assert min_long < 0
    assert max_long > 0

    extents = (min_long, min_lat, max_long, max_lat)
    gen = dsm.copernicus_tiles_latlon_covering_extents(extents)
    latitudes, longitudes = get_unique_lat_longs_from_coord_pairs(gen)

    assert latitudes == {-36, -35, -34}
    assert longitudes == {-180, -179, -178, 179}
