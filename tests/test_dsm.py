# Partial tests for the DSM/Digital Surface Model module
import pytest

from wagl import dsm


@pytest.fixture
def l9_antarctic_volcano_extents():
    # order is min_x, min_y, max_x, max_y
    return -119.9633131, -77.2715533, -108.62444196, -74.51953282


@pytest.fixture
def l9_buenos_aires_extents():
    # order is min_x, min_y, max_x, max_y
    return -60.05997761, -35.66956971, -57.50299335, -33.54584702


@pytest.fixture
def l9_italy_extents():
    # from LC09_L1TP_188033_20231030_20231030_02_T1.tar
    return 14.491808, 37.828106, 17.192353, 39.956304


def get_lat_longs_from_coord_pairs(tuple_sequence):
    latitudes, longitudes = zip(*tuple_sequence)
    return set(latitudes), set(longitudes)


def test_calculate_latitude_extent_southern_hemisphere(l9_antarctic_volcano_extents):
    gen = dsm.copernicus_tiles_latlon_covering_geobox(l9_antarctic_volcano_extents)
    latitudes, longitudes = get_lat_longs_from_coord_pairs(gen)

    assert latitudes == {-78, -77, -76, -75}
    assert longitudes == set(range(-120, -108))


def test_calculate_latitude_extent_southern_hemisphere2(l9_buenos_aires_extents):
    gen = dsm.copernicus_tiles_latlon_covering_geobox(l9_buenos_aires_extents)
    latitudes, longitudes = get_lat_longs_from_coord_pairs(gen)

    assert latitudes == {-34, -35, -36}
    assert longitudes == {-61, -60, -59, -58}


def test_calculate_latitude_extent_northern_hemisphere(l9_italy_extents):
    gen = dsm.copernicus_tiles_latlon_covering_geobox(l9_italy_extents)
    latitudes, longitudes = get_lat_longs_from_coord_pairs(gen)

    assert latitudes == {37, 38, 39}
    assert longitudes == {14, 15, 16, 17}
