import pathlib

from wagl import modtran


def test_get_solar_angles():
    module_path = pathlib.Path(__file__)
    parent = module_path.parent
    tp6_path = parent / "data/POINT-8-ALBEDO-0-solar_zenith_cropped.tp6"

    solar_zenith = modtran._get_solar_angles(tp6_path)
    assert solar_zenith.shape == (35,)
    assert solar_zenith[0] == "53.76652817"
    assert solar_zenith[1] == "53.76353292"
