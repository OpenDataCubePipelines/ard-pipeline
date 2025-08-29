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
def l9_buenos_aires_extents():
    # Extents converted from LC09_L1TP_225084_20241222_20241222_02_T1.tar

    # order is min_x, min_y, max_x, max_y
    return -60.05997761, -35.66956971, -57.50299335, -33.54584702


@pytest.fixture
def l9_italy_extents():
    # from LC09_L1TP_188033_20231030_20231030_02_T1.tar
    return 14.491808, 37.828106, 17.192353, 39.956304


@pytest.fixture
def l5_wagga_extents():
    # From LT05_L1TP_092084_20080925_20161029_01_T1.tar
    return 145.434996, -35.581614, 148.077655, -33.653483


# Section: helper functions


def get_unique_lat_longs_from_coord_pairs(tuple_sequence):
    """
    Split (lat, long) tuple sequence into separate latitude/longitude sequences.
    """
    latitudes, longitudes = zip(*tuple_sequence)
    return set(latitudes), set(longitudes)


# Section: basic extents testing


def test_lat_long_extents_southern_hemisphere_polar(l9_antarctic_volcano_extents):
    # Ensure CopDEM tile search provides correct coverage for Landsat scene extents
    gen = dsm.copernicus_tiles_latlon_covering_geobox(l9_antarctic_volcano_extents)
    latitudes, longitudes = get_unique_lat_longs_from_coord_pairs(gen)

    assert latitudes == {-78, -77, -76, -75}
    assert longitudes == set(range(-120, -108))


def test_lat_long_extents_southern_and_western_hemispheres(l9_buenos_aires_extents):
    gen = dsm.copernicus_tiles_latlon_covering_geobox(l9_buenos_aires_extents)
    latitudes, longitudes = get_unique_lat_longs_from_coord_pairs(gen)

    assert latitudes == {-34, -35, -36}
    assert longitudes == {-61, -60, -59, -58}


def test_lat_long_extents_northern_and_western_hemispheres(l9_italy_extents):
    gen = dsm.copernicus_tiles_latlon_covering_geobox(l9_italy_extents)
    latitudes, longitudes = get_unique_lat_longs_from_coord_pairs(gen)

    assert latitudes == {37, 38, 39}
    assert longitudes == {14, 15, 16, 17}


def test_lat_long_extents_southern_and_eastern_hemispheres(l5_wagga_extents):
    gen = dsm.copernicus_tiles_latlon_covering_geobox(l5_wagga_extents)
    latitudes, longitudes = get_unique_lat_longs_from_coord_pairs(gen)

    # these lat/long extents confirmed by loading CopDEM tiles in QGIS
    assert latitudes == {-36, -35, -34}
    assert longitudes == {145, 146, 147, 148}


# Section: more detailed lat/long tests
# confirm the exact (lat, long) extents pairs for covering tiles


def test_eastern_hemisphere_lat_long_pairs(l5_wagga_extents):
    gen = dsm.copernicus_tiles_latlon_covering_geobox(l5_wagga_extents)

    for lat in (-36, -35, -34):
        for long in (145, 146, 147, 148):
            assert (lat, long) == gen.__next__()


def test_western_hemisphere_lat_long_pairs(l9_antarctic_volcano_extents):
    gen = dsm.copernicus_tiles_latlon_covering_geobox(l9_antarctic_volcano_extents)

    for lat in (-78, -77, -76, -75):
        for long in range(-120, -108):
            assert (lat, long) == gen.__next__()
