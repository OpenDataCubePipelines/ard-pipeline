#!/usr/bin/env python

"""Test several different configurations of the tp5 file format."""

import re
import unittest
from os.path import abspath, dirname
from os.path import join as pjoin

from wagl import modtran_profiles as mp

DATA_DIR = pjoin(dirname(abspath(__file__)), "data")
FNAME1 = pjoin(DATA_DIR, "TL_alb_0.tp5")
FNAME2 = pjoin(DATA_DIR, "TL_alb_0_binary.tp5")
FNAME3 = pjoin(DATA_DIR, "TL_alb_t.tp5")
FNAME4 = pjoin(DATA_DIR, "TL_alb_t_binary.tp5")
FNAME5 = pjoin(DATA_DIR, "point-3-albedo-0.tp5")
FNAME6 = pjoin(DATA_DIR, "point-3-albedo-0_binary.tp5")
FNAME7 = pjoin(DATA_DIR, "point-3-albedo-t.tp5")
FNAME8 = pjoin(DATA_DIR, "point-3-albedo-t_binary.tp5")


def assert_similar_output(input_file: str, expected_contents: str):
    with open(input_file) as src:
        data = "".join(src.readlines())

    expected_content = remove_duplicate_whitespace(expected_contents)
    data_clean = remove_duplicate_whitespace(data)
    assert expected_content == data_clean


def remove_duplicate_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


class Tp5Test(unittest.TestCase):
    """Test different configurations of the tp5 file format,
    and that the input/output round trips.
    """

    def test_midlat_summer_albedo(self):
        """Test the mid latitude summer albedo configuration."""
        test = mp.MIDLAT_SUMMER_ALBEDO.format(
            binary=" ",
            albedo=0.0,
            water=1.07000122070313,
            ozone=0.28499999642372,
            filter_function="landsat7_vsir.flt",
            visibility=-0.02264800000000,
            elevation=0.70900000000000,
            sat_height=705.0,
            sat_view=171.000748,
            doy=212,
            lat=-29.33856209871443,
            lon=209.88857485506449,
            time=23.73920805027778,
            sat_azimuth=279.408417,
        )

        assert_similar_output(FNAME1, test)

    def test_midlat_summer_albedo_b(self):
        """Test the mid latitude summer albedo binary configuration."""
        test = mp.MIDLAT_SUMMER_ALBEDO.format(
            binary="T",
            albedo=0.0,
            water=1.07000122070313,
            ozone=0.28499999642372,
            filter_function="landsat7_vsir.flt",
            visibility=-0.02264800000000,
            elevation=0.70900000000000,
            sat_height=705.0,
            sat_view=171.000748,
            doy=212,
            lat=-29.33856209871443,
            lon=209.88857485506449,
            time=23.73920805027778,
            sat_azimuth=279.408417,
        )

        assert_similar_output(FNAME2, test)

    def test_midlat_summer_trans(self):
        """Test the mid latitude summer transmittance configuration."""
        test = mp.MIDLAT_SUMMER_TRANSMITTANCE.format(
            binary=" ",
            albedo=0.0,
            water=1.07000122070313,
            ozone=0.28499999642372,
            filter_function="landsat7_vsir.flt",
            visibility=-0.02264800000000,
            elevation=0.70900000000000,
            sat_height=705.0,
            sat_view=171.000748,
            doy=212,
            sat_view_offset=180.0 - 171.000748,
        )

        assert_similar_output(FNAME3, test)

    def test_midlat_summer_trans_b(self):
        """Test the mid latitude summer transmittance binary configuration."""
        test = mp.MIDLAT_SUMMER_TRANSMITTANCE.format(
            binary="T",
            albedo=0.0,
            water=1.07000122070313,
            ozone=0.28499999642372,
            filter_function="landsat7_vsir.flt",
            visibility=-0.02264800000000,
            elevation=0.70900000000000,
            sat_height=705.0,
            sat_view=171.000748,
            doy=212,
            sat_view_offset=180.0 - 171.000748,
        )

        assert_similar_output(FNAME4, test)

    def test_tropical_albedo(self):
        """Test the tropical albedo configuration."""
        test = mp.TROPICAL_ALBEDO.format(
            binary=" ",
            albedo=0.0,
            water=1.3500000000000001,
            ozone=0.25600001,
            filter_function="landsat8_vsir.flt",
            visibility=-0.043157435953617096,
            elevation=0.378,
            sat_height=705.0,
            sat_view=171.00043,
            doy=200,
            lat=-20.247597626228778,
            lon=229.23617402910139,
            time=1.2133087877777777,
            sat_azimuth=278.77069,
        )

        assert_similar_output(FNAME5, test)

    def test_tropical_albedo_b(self):
        """Test the tropical albedo binary configuration."""
        test = mp.TROPICAL_ALBEDO.format(
            binary="T",
            albedo=0.0,
            water=1.3500000000000001,
            ozone=0.25600001,
            filter_function="landsat8_vsir.flt",
            visibility=-0.043157435953617096,
            elevation=0.378,
            sat_height=705.0,
            sat_view=171.00043,
            doy=200,
            lat=-20.247597626228778,
            lon=229.23617402910139,
            time=1.2133087877777777,
            sat_azimuth=278.77069,
        )

        assert_similar_output(FNAME6, test)

    def test_tropical_trans(self):
        """Test the tropical transmittance configuration."""
        test = mp.TROPICAL_TRANSMITTANCE.format(
            binary=" ",
            albedo=0.0,
            water=1.3500000000000001,
            ozone=0.25600001,
            filter_function="landsat8_vsir.flt",
            visibility=-0.043157435953617096,
            elevation=0.378,
            sat_height=705.0,
            sat_view=171.00043,
            doy=200,
            sat_view_offset=8.99957275390625,
        )

        assert_similar_output(FNAME7, test)

    def test_tropical_trans_b(self):
        """Test the tropical transmittance binary configuration."""
        test = mp.TROPICAL_TRANSMITTANCE.format(
            binary="T",
            albedo=0.0,
            water=1.3500000000000001,
            ozone=0.25600001,
            filter_function="landsat8_vsir.flt",
            visibility=-0.043157435953617096,
            elevation=0.378,
            sat_height=705.0,
            sat_view=171.00043,
            doy=200,
            sat_view_offset=8.99957275390625,
        )

        assert_similar_output(FNAME8, test)

    def test_sbt(self):
        """Test the surface brightness temperature configuration."""
        pass


if __name__ == "__main__":
    unittest.main()
