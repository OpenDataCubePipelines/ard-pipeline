import datetime
import unittest
from os.path import abspath, dirname
from os.path import join as pjoin

from wagl.mtl import load_mtl, parse_type

DATA_DIR = pjoin(dirname(abspath(__file__)), "data")

L5_MTL1 = pjoin(DATA_DIR, "LANDSAT5", "L5090081_08120090407_MTL.txt")
L5_MTL2 = pjoin(
    DATA_DIR, "LANDSAT5", "LT05_L1TP_095066_20100601_20170222_01_T1_MTL.txt"
)
L7_MTL1 = pjoin(DATA_DIR, "LANDSAT7", "L71090081_08120090415_MTL.txt")
L7_MTL2 = pjoin(
    DATA_DIR, "LANDSAT7", "LE07_L1TP_112066_20020218_20170221_01_T1_MTL.txt"
)
L7_MTLRTC2 = pjoin(
    DATA_DIR,
    "LANDSAT7",
    "LE71140812021051EDC00__C2_RT",
    "LE07_L1TP_114081_20210220_20210220_02_RT_MTL.txt",
)
L8_MTL1 = pjoin(DATA_DIR, "LANDSAT8", "LO80900842013284ASA00_MTL.txt")
L8_MTL2 = pjoin(DATA_DIR, "LANDSAT8", "LO80900842013284ASA00_MTL.txt")
L8_MTL1C2 = pjoin(
    DATA_DIR, "LANDSAT8", "LC08_L1TP_092084_20201029_20201106_02_T1_MTL.txt"
)
L8_MTRTC2 = pjoin(
    DATA_DIR,
    "LANDSAT8",
    "LC81060632021051LGN00__C2_RT",
    "LC08_L1TP_106063_20210220_20210220_02_RT_MTL.txt",
)


class TypeParserTest(unittest.TestCase):
    def test_integer(self):
        num = parse_type("1")
        assert num == 1

    def test_float(self):
        num = parse_type("1.0")
        assert num == 1.0

    def test_datetime(self):
        dt0 = parse_type("2013-11-07T01:42:41Z")
        dt1 = datetime.datetime(2013, 11, 7, 1, 42, 41)
        assert dt0 == dt1

    def test_quoted_datetime(self):
        dt0 = parse_type('"2013-11-07T01:42:41Z"')
        dt1 = datetime.datetime(2013, 11, 7, 1, 42, 41)
        assert dt0 == dt1

    def test_date(self):
        dt0 = parse_type("2013-11-07")
        dt1 = datetime.date(2013, 11, 7)
        assert dt0 == dt1

    def test_time(self):
        dt0 = parse_type("23:46:09.1442826Z")
        dt1 = datetime.time(23, 46, 9, 144282)
        assert dt0 == dt1

    def test_quoted_time(self):
        dt0 = parse_type('"23:46:09.1442826Z"')
        dt1 = datetime.time(23, 46, 9, 144282)
        assert dt0 == dt1

    def test_yes(self):
        resp = parse_type("Y")
        assert resp is True

    def test_no(self):
        resp = parse_type("N")
        assert resp is False

    def test_none(self):
        val = parse_type("NONE")
        assert val is None

    def test_str(self):
        s = parse_type("1adsd")
        assert s == "1adsd"


class Landsat5MTL1ParserTest(unittest.TestCase):
    def test_load(self):
        tree = load_mtl(L5_MTL1)
        assert len(tree) == 9
        assert "METADATA_FILE_INFO" in tree
        assert "PRODUCT_METADATA" in tree
        assert "MIN_MAX_RADIANCE" in tree
        assert "MIN_MAX_PIXEL_VALUE" in tree
        assert "PRODUCT_PARAMETERS" in tree
        assert "CORRECTIONS_APPLIED" in tree
        assert "PROJECTION_PARAMETERS" in tree
        assert "UTM_PARAMETERS" in tree


class Landsat5MTL2ParserTest(unittest.TestCase):
    def test_load(self):
        tree = load_mtl(L5_MTL2)
        assert len(tree) == 10
        assert "METADATA_FILE_INFO" in tree
        assert "PRODUCT_METADATA" in tree
        assert "MIN_MAX_RADIANCE" in tree
        assert "MIN_MAX_REFLECTANCE" in tree
        assert "MIN_MAX_PIXEL_VALUE" in tree
        assert "PRODUCT_PARAMETERS" in tree
        assert "PROJECTION_PARAMETERS" in tree
        assert "IMAGE_ATTRIBUTES" in tree
        assert "THERMAL_CONSTANTS" in tree


class Landsat7MTL1ParserTest(unittest.TestCase):
    def test_load(self):
        tree = load_mtl(L7_MTL1)
        assert len(tree) == 8
        assert "METADATA_FILE_INFO" in tree
        assert "PRODUCT_METADATA" in tree
        assert "MIN_MAX_RADIANCE" in tree
        assert "MIN_MAX_PIXEL_VALUE" in tree
        assert "PRODUCT_PARAMETERS" in tree
        assert "CORRECTIONS_APPLIED" in tree
        assert "PROJECTION_PARAMETERS" in tree
        assert "UTM_PARAMETERS" in tree


class Landsat7MTL2ParserTest(unittest.TestCase):
    def test_load(self):
        tree = load_mtl(L7_MTL2)
        assert len(tree) == 10
        assert "METADATA_FILE_INFO" in tree
        assert "PRODUCT_METADATA" in tree
        assert "MIN_MAX_RADIANCE" in tree
        assert "MIN_MAX_REFLECTANCE" in tree
        assert "MIN_MAX_PIXEL_VALUE" in tree
        assert "PRODUCT_PARAMETERS" in tree
        assert "PROJECTION_PARAMETERS" in tree
        assert "IMAGE_ATTRIBUTES" in tree
        assert "THERMAL_CONSTANTS" in tree


class Landsat8MTL1ParserTest(unittest.TestCase):
    def test_load(self):
        tree = load_mtl(L8_MTL1)
        assert len(tree) == 9
        assert "METADATA_FILE_INFO" in tree
        assert "PRODUCT_METADATA" in tree
        assert "IMAGE_ATTRIBUTES" in tree
        assert "MIN_MAX_RADIANCE" in tree
        assert "MIN_MAX_REFLECTANCE" in tree
        assert "MIN_MAX_PIXEL_VALUE" in tree
        assert "RADIOMETRIC_RESCALING" in tree
        assert "TIRS_THERMAL_CONSTANTS" in tree
        assert "PROJECTION_PARAMETERS" in tree


class Landsat8MTL1C2ParserTest(unittest.TestCase):
    def test_load(self):
        tree = load_mtl(L8_MTL1C2)
        assert len(tree) == 10
        assert "PRODUCT_CONTENTS" in tree  # was METADATA_FILE_INFO in C1
        assert "IMAGE_ATTRIBUTES" in tree  # C1 PRODUCT_METADATA info added here in C2
        assert "LEVEL1_MIN_MAX_RADIANCE" in tree
        assert "LEVEL1_MIN_MAX_REFLECTANCE" in tree
        assert "LEVEL1_MIN_MAX_PIXEL_VALUE" in tree
        assert "LEVEL1_RADIOMETRIC_RESCALING" in tree
        assert "LEVEL1_THERMAL_CONSTANTS" in tree
        assert "LEVEL1_PROJECTION_PARAMETERS" in tree


class Landsat8MTRTC2ParserTest(unittest.TestCase):
    def test_load(self):
        tree = load_mtl(L8_MTRTC2)
        assert len(tree) == 10
        assert "PRODUCT_CONTENTS" in tree
        assert "IMAGE_ATTRIBUTES" in tree
        assert "LEVEL1_MIN_MAX_RADIANCE" in tree
        assert "LEVEL1_MIN_MAX_REFLECTANCE" in tree
        assert "LEVEL1_MIN_MAX_PIXEL_VALUE" in tree
        assert "LEVEL1_RADIOMETRIC_RESCALING" in tree
        assert "LEVEL1_THERMAL_CONSTANTS" in tree
        assert "LEVEL1_PROJECTION_PARAMETERS" in tree


class Landsat7MTRTC2ParserTest(unittest.TestCase):
    def test_load(self):
        tree = load_mtl(L7_MTLRTC2)
        assert len(tree) == 11
        assert "PRODUCT_CONTENTS" in tree
        assert "IMAGE_ATTRIBUTES" in tree
        assert "LEVEL1_MIN_MAX_RADIANCE" in tree
        assert "LEVEL1_MIN_MAX_REFLECTANCE" in tree
        assert "LEVEL1_MIN_MAX_PIXEL_VALUE" in tree
        assert "LEVEL1_RADIOMETRIC_RESCALING" in tree
        assert "LEVEL1_THERMAL_CONSTANTS" in tree
        assert "LEVEL1_PROJECTION_PARAMETERS" in tree


class Landsat8MTL2ParserTest(unittest.TestCase):
    def test_load(self):
        tree = load_mtl(L8_MTL2)
        assert len(tree) == 9
        assert "METADATA_FILE_INFO" in tree
        assert "PRODUCT_METADATA" in tree
        assert "IMAGE_ATTRIBUTES" in tree
        assert "MIN_MAX_RADIANCE" in tree
        assert "MIN_MAX_REFLECTANCE" in tree
        assert "MIN_MAX_PIXEL_VALUE" in tree
        assert "RADIOMETRIC_RESCALING" in tree
        assert "TIRS_THERMAL_CONSTANTS" in tree
        assert "PROJECTION_PARAMETERS" in tree


if __name__ == "__main__":
    unittest.main()
