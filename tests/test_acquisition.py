import datetime
import unittest
from os.path import abspath, dirname
from os.path import join as pjoin

from gaip.acquisition import acquisitions
from gaip.acquisition.landsat import Landsat8Acquisition, LandsatAcquisition
from gaip.constants import BandType
from gaip.temperature import temperature_at_sensor

DATA_DIR = pjoin(dirname(abspath(__file__)), "data")

LS5_SCENE1 = pjoin(DATA_DIR, "LANDSAT5", "LS5_TM_OTH_P51_GALPGS01-002_090_081_20090407")
LS7_SCENE1 = pjoin(
    DATA_DIR, "LANDSAT7", "LS7_ETM_OTH_P51_GALPGS01-002_090_081_20090415"
)
LS8_SCENE1 = pjoin(
    DATA_DIR, "LANDSAT8", "LS8_OLITIRS_OTH_P51_GALPGS01-032_090_084_20131011"
)


class AcquisitionLoadMtlTest(unittest.TestCase):
    def test_load_acquisitions_ls5_scene1(self):
        acq = acquisitions(LS5_SCENE1).get_acquisitions()
        assert len(acq) == 7

    def test_load_acquisitions_ls7_scene1(self):
        acq = acquisitions(LS7_SCENE1).get_acquisitions()
        assert len(acq) == 8

    def test_load_acquisitions_ls8_scene1(self):
        acq = acquisitions(LS8_SCENE1).get_acquisitions()
        assert len(acq) == 9


class AcquisitionsContainerTest(unittest.TestCase):
    def test_groups_ls5_scene1(self):
        scene = acquisitions(LS5_SCENE1)
        assert len(scene.groups) == 1

    def test_groups_ls7_scene1(self):
        scene = acquisitions(LS7_SCENE1)
        assert len(scene.groups) == 1

    def test_groups_ls8_scene1(self):
        scene = acquisitions(LS8_SCENE1)
        assert len(scene.groups) == 1

    def test_granules_ls5_scene1(self):
        scene = acquisitions(LS5_SCENE1)
        assert scene.granules[0] is None

    def test_granules_ls7_scene1(self):
        scene = acquisitions(LS7_SCENE1)
        assert scene.granules[0] is None

    def test_granules_ls8_scene1(self):
        scene = acquisitions(LS8_SCENE1)
        assert scene.granules[0] is None


class Landsat5Scene1AcquisitionTest(unittest.TestCase):
    def setUp(self):
        self.acqs = acquisitions(LS5_SCENE1).get_acquisitions()

    def test_type(self):
        for acq in self.acqs:
            assert isinstance(acq, LandsatAcquisition)

    def test_band_type(self):
        assert self.acqs[0].band_type == BandType.Reflective
        assert self.acqs[1].band_type == BandType.Reflective
        assert self.acqs[2].band_type == BandType.Reflective
        assert self.acqs[3].band_type == BandType.Reflective
        assert self.acqs[4].band_type == BandType.Reflective
        assert self.acqs[5].band_type == BandType.Thermal
        assert self.acqs[6].band_type == BandType.Reflective

    def test_acquisition_datetime(self):
        for acq in self.acqs:
            assert acq.acquisition_datetime == datetime.datetime(
                2009, 4, 7, 23, 36, 9, 88050
            )

    def test_min_radiance_band1(self):
        assert self.acqs[0].min_radiance == -1.52

    def test_max_radiance_band1(self):
        assert self.acqs[0].max_radiance == 193.0

    def test_min_quantize_band1(self):
        assert self.acqs[0].min_quantize == 1.0

    def test_max_quantize_band1(self):
        assert self.acqs[0].max_quantize == 255.0

    def test_solar_azimuth(self):
        assert self.acqs[0].solar_azimuth == 48.17689881

    def test_solar_elevation(self):
        assert self.acqs[0].solar_elevation == 39.40143058

    def test_gain(self):
        self.assertAlmostEqual(self.acqs[0].gain, 0.7658267716535433)

    def test_bias(self):
        self.assertAlmostEqual(self.acqs[0].bias, -2.2858267716535465)

    def test_sensor_id(self):
        for acq in self.acqs:
            assert acq.sensor_id == "TM"

    def test_platform_id(self):
        for acq in self.acqs:
            assert acq.platform_id == "LANDSAT_5"

    def test_samples(self):
        assert self.acqs[0].samples == 95

    def test_lines(self):
        assert self.acqs[0].lines == 83

    def test_read(self):
        assert self.acqs[0].data()[70, 30] == 65

    def test_spectral_filter_file_vsir(self):
        assert self.acqs[0].spectral_filter_file == "landsat5_vsir.flt"

    def test_spectral_filter_file_thermal(self):
        assert self.acqs[5].spectral_filter_file == "landsat5_thermal.flt"

    def test_temperature(self):
        result = temperature_at_sensor(self.acqs[5], window=((40, 41), (40, 41)))
        self.assertAlmostEqual(result[0, 0], 292.87979272)


class Landsat7Mtl1AcquisitionTest(unittest.TestCase):
    def setUp(self):
        self.acqs = acquisitions(LS7_SCENE1).get_acquisitions()

    def test_type(self):
        for acq in self.acqs:
            assert isinstance(acq, LandsatAcquisition)

    def test_band_type(self):
        assert self.acqs[0].band_type == BandType.Reflective
        assert self.acqs[1].band_type == BandType.Reflective
        assert self.acqs[2].band_type == BandType.Reflective
        assert self.acqs[3].band_type == BandType.Reflective
        assert self.acqs[4].band_type == BandType.Reflective
        assert self.acqs[5].band_type == BandType.Thermal
        assert self.acqs[6].band_type == BandType.Thermal
        assert self.acqs[7].band_type == BandType.Reflective

    def test_acquisition_datetime(self):
        for acq in self.acqs:
            assert acq.acquisition_datetime == datetime.datetime(
                2009, 4, 15, 23, 39, 26, 931462
            )

    def test_min_radiance_band1(self):
        assert self.acqs[0].min_radiance == -6.2

    def test_max_radiance_band1(self):
        assert self.acqs[0].max_radiance == 191.6

    def test_min_quantize_band1(self):
        assert self.acqs[0].min_quantize == 1.0

    def test_max_quantize_band1(self):
        assert self.acqs[0].max_quantize == 255.0

    def test_solar_azimuth(self):
        assert self.acqs[0].solar_azimuth == 44.50200305

    def test_solar_elevation(self):
        assert self.acqs[0].solar_elevation == 37.94917208

    def test_gain(self):
        self.assertAlmostEqual(self.acqs[0].gain, 0.7787401574803149)

    def test_bias(self):
        self.assertAlmostEqual(self.acqs[0].bias, -6.978740157480303)

    def test_sensor_id(self):
        for acq in self.acqs:
            assert acq.sensor_id == "ETM+"

    def test_platform_id(self):
        for acq in self.acqs:
            assert acq.platform_id == "LANDSAT_7"

    def test_samples(self):
        assert self.acqs[0].samples == 96

    def test_lines(self):
        assert self.acqs[0].lines == 83

    def test_read(self):
        assert self.acqs[0].data()[70, 30] == 61

    def test_spectral_filter_file_vsir(self):
        assert self.acqs[0].spectral_filter_file == "landsat7_vsir.flt"

    def test_spectral_filter_file_thermal(self):
        assert self.acqs[5].spectral_filter_file == "landsat7_thermal.flt"

    def test_temperature61(self):
        result = temperature_at_sensor(self.acqs[5], window=((41, 42), (41, 42)))
        self.assertAlmostEqual(result[0, 0], 297.00875604)

    def test_temperature62(self):
        result = temperature_at_sensor(self.acqs[6], window=((41, 42), (41, 42)))
        self.assertAlmostEqual(result[0, 0], 297.3971409)


class Landsat8Mtl1AcquisitionTest(unittest.TestCase):
    def setUp(self):
        self.acqs = acquisitions(LS8_SCENE1).get_acquisitions()

    def test_type(self):
        for acq in self.acqs:
            assert isinstance(acq, Landsat8Acquisition)

    def test_band_type(self):
        assert self.acqs[0].band_type == BandType.Reflective
        assert self.acqs[1].band_type == BandType.Thermal
        assert self.acqs[2].band_type == BandType.Thermal
        assert self.acqs[3].band_type == BandType.Reflective
        assert self.acqs[4].band_type == BandType.Reflective
        assert self.acqs[5].band_type == BandType.Reflective
        assert self.acqs[6].band_type == BandType.Reflective
        assert self.acqs[6].band_type == BandType.Reflective
        assert self.acqs[6].band_type == BandType.Reflective

    def test_acquisition_datetime(self):
        for acq in self.acqs:
            assert acq.acquisition_datetime == datetime.datetime(
                2013, 10, 11, 23, 52, 10, 570334
            )

    def test_min_radiance_band1(self):
        assert self.acqs[0].min_radiance == -63.00884

    def test_max_radiance_band1(self):
        assert self.acqs[0].max_radiance == 763.00067

    def test_min_quantize_band1(self):
        assert self.acqs[0].min_quantize == 1.0

    def test_max_quantize_band1(self):
        assert self.acqs[0].max_quantize == 65535

    def test_solar_azimuth(self):
        assert self.acqs[0].solar_azimuth == 50.86391564

    def test_solar_elevation(self):
        assert self.acqs[0].solar_elevation == 52.04105874

    def test_gain(self):
        self.assertAlmostEqual(self.acqs[0].gain, 0.0126, 4)

    def test_bias(self):
        self.assertAlmostEqual(self.acqs[0].bias, -63.0214, 4)

    def test_sensor_id(self):
        for acq in self.acqs:
            assert acq.sensor_id == "OLI"

    def test_platform_id(self):
        for acq in self.acqs:
            assert acq.platform_id == "LANDSAT_8"

    def test_samples(self):
        assert self.acqs[0].samples == 94

    def test_lines(self):
        assert self.acqs[0].lines == 95

    def test_read(self):
        assert self.acqs[0].data()[70, 30] == 11003

    def test_spectral_filter_file_vsir(self):
        assert self.acqs[0].spectral_filter_file == "landsat8_vsir.flt"

    def test_spectral_filter_file_thermal(self):
        assert self.acqs[1].spectral_filter_file == "landsat8_thermal.flt"

    def test_temperature10(self):
        result = temperature_at_sensor(self.acqs[1], window=((41, 42), (41, 42)))
        self.assertAlmostEqual(result[0, 0], 293.63805603)

    def test_temperature11(self):
        result = temperature_at_sensor(self.acqs[2], window=((41, 42), (41, 42)))
        self.assertAlmostEqual(result[0, 0], 292.90268541)


if __name__ == "__main__":
    unittest.main()