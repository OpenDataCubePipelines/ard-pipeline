#!/usr/bin/env python
import re
import tempfile
import unittest
from os.path import abspath, dirname
from os.path import join as pjoin

from wagl.modtran_profiles import MIDLAT_SUMMER_ALBEDO, MIDLAT_SUMMER_TRANSMITTANCE

DATA_DIR = pjoin(dirname(abspath(__file__)), "data")


def assert_similar_str(input1: str, input2: str):
    input1 = remove_duplicate_whitespace(input1)
    input2 = remove_duplicate_whitespace(input2)
    assert input1 == input2


def remove_duplicate_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


class Tp5ReformatTest(unittest.TestCase):
    """Test that the string formatting matches the old
    output method.
    """

    def test_midlat_summer_albdo0(self):
        """Test a mid latitude summer albedo 0 profile."""
        self.maxDiff = None

        with open(pjoin(DATA_DIR, "TL_alb_0.tp5")) as src:
            ref_albedo = "".join(src.readlines())

        with tempfile.TemporaryDirectory() as tmpdir:
            out_fname = pjoin(tmpdir, "test-midlat-summer.tp5")
            with open(out_fname, "w") as src:
                kwargs = {
                    "albedo": 0.0,
                    "water": 1.07000122070313,
                    "ozone": 0.28499999642372,
                    "filter_function": "landsat7_vsir.flt",
                    "visibility": -0.02264800000000,
                    "elevation": 0.70900000000000,
                    "sat_height": 705.0,
                    "sat_view": 171.000748,
                    "doy": 212,
                    "lat": -29.33856209871443,
                    "lon": 209.88857485506449,
                    "time": 23.73920805027778,
                    "sat_azimuth": 279.408417,
                    "binary": " ",
                }

                src.write(MIDLAT_SUMMER_ALBEDO.format(**kwargs))

            with open(out_fname) as src:
                test_albedo = "".join(src.readlines())

        assert_similar_str(test_albedo, ref_albedo)

    def test_midlat_summer_transmittance(self):
        """Test a mid latitude summer transmittance profile."""
        self.maxDiff = None

        with open(pjoin(DATA_DIR, "TL_alb_t.tp5")) as src:
            ref_trans = "".join(src.readlines())

        with tempfile.TemporaryDirectory() as tmpdir:
            out_fname = pjoin(tmpdir, "test-midlat-summer-trans.tp5")
            with open(out_fname, "w") as src:
                kwargs = {
                    "albedo": 0.0,
                    "water": 1.07000122070313,
                    "ozone": 0.28499999642372,
                    "filter_function": "landsat7_vsir.flt",
                    "visibility": -0.02264800000000,
                    "elevation": 0.70900000000000,
                    "sat_height": 705.0,
                    "sat_view": 171.000748,
                    "doy": 212,
                    "sat_view_offset": 180.0 - 171.000748,
                    "binary": " ",
                }

                src.write(MIDLAT_SUMMER_TRANSMITTANCE.format(**kwargs))

            with open(out_fname) as src:
                test_trans = "".join(src.readlines())

        assert_similar_str(test_trans, ref_trans)


if __name__ == "__main__":
    unittest.main()
