#!/usr/bin/env python


import unittest

from data import LS7_GAP_MASK, LS7_NO_GAP_MASK

from wagl.acquisition import acquisitions


class GapMaskRadianceTest(unittest.TestCase):
    """Test that the SLC gap mask loads correctly.
    The values to check against were derived by manually selecting
    the band and the corresponding gap mask, and creating a null mask.
    """

    def setUp(self):
        self.acqs = acquisitions(LS7_GAP_MASK).get_all_acquisitions()

    def test_band8(self):
        acq = self.acqs[0]
        mask = acq.radiance_data() == -999
        count = mask.sum()
        assert count == 259512

    def test_band1(self):
        acq = self.acqs[1]
        mask = acq.radiance_data() == -999
        count = mask.sum()
        assert count == 64746

    def test_band2(self):
        acq = self.acqs[2]
        mask = acq.radiance_data() == -999
        count = mask.sum()
        assert count == 64766

    def test_band3(self):
        acq = self.acqs[3]
        mask = acq.radiance_data() == -999
        count = mask.sum()
        assert count == 64761

    def test_band4(self):
        acq = self.acqs[4]
        mask = acq.radiance_data() == -999
        count = mask.sum()
        assert count == 64770

    def test_band5(self):
        acq = self.acqs[5]
        mask = acq.radiance_data() == -999
        count = mask.sum()
        assert count == 64769

    def test_band61(self):
        acq = self.acqs[6]
        mask = acq.radiance_data() == -999
        count = mask.sum()
        assert count == 64862

    def test_band62(self):
        acq = self.acqs[7]
        mask = acq.radiance_data() == -999
        count = mask.sum()
        assert count == 64898

    def test_band7(self):
        acq = self.acqs[8]
        mask = acq.radiance_data() == -999
        count = mask.sum()
        assert count == 64747


class NoGapMaskRadianceTest(unittest.TestCase):
    """Test that the abscence of a gap mask has no effect on loading
    the data.
    """

    def setUp(self):
        self.acqs = acquisitions(LS7_NO_GAP_MASK).get_all_acquisitions()

    def test_band8(self):
        acq = self.acqs[0]
        _ = acq.radiance_data()
        count = acq._gap_mask.sum()
        assert count == 0

    def test_band1(self):
        acq = self.acqs[1]
        _ = acq.radiance_data()
        count = acq._gap_mask.sum()
        assert count == 0

    def test_band2(self):
        acq = self.acqs[2]
        _ = acq.radiance_data()
        count = acq._gap_mask.sum()
        assert count == 0

    def test_band3(self):
        acq = self.acqs[3]
        _ = acq.radiance_data()
        count = acq._gap_mask.sum()
        assert count == 0

    def test_band4(self):
        acq = self.acqs[4]
        _ = acq.radiance_data()
        count = acq._gap_mask.sum()
        assert count == 0

    def test_band5(self):
        acq = self.acqs[5]
        _ = acq.radiance_data()
        count = acq._gap_mask.sum()
        assert count == 0

    def test_band61(self):
        acq = self.acqs[6]
        _ = acq.radiance_data()
        count = acq._gap_mask.sum()
        assert count == 0

    def test_band62(self):
        acq = self.acqs[7]
        _ = acq.radiance_data()
        count = acq._gap_mask.sum()
        assert count == 0

    def test_band7(self):
        acq = self.acqs[8]
        _ = acq.radiance_data()
        count = acq._gap_mask.sum()
        assert count == 0
