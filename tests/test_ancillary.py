from unittest import mock

import pandas as pd

from wagl import (
    acquisition,
    ancillary,
    era5,
)


def test_collect_era5_ancillary():
    # Loose test to ensure ERA5 collection workflow runs end to end. Underlying
    # I/O functionality is skipped with mocks. This is not ideal, but is a problem
    # with the s/w architecture burying I/O dependencies in the code.
    #
    # A secondary, related problem is that this test is *coupled* to internal
    # details of era5.profile_data_frame_workflow(). If the workflow changes,
    # it's possible this test will require changes
    acq = mock.MagicMock(acquisition.Acquisition)
    lat_longs = ((-35, 149), (-35.5, 149.5), (-36, 150))
    out_group = mock.MagicMock()
    era5_data_dir = "fake-dir"

    with (
        mock.patch(
            "wagl.era5.profile_data_frame_workflow"
        ) as m_profile_data_frame_workflow,
        mock.patch("wagl.ancillary.write_dataframe") as m_write_dataframe,
        mock.patch("wagl.ancillary.write_scalar") as m_write_scalar,
        mock.patch(
            "wagl.era5.ozone_workflow", return_value=[0.02, 0.025, 0.03]
        ) as m_ozone_workflow,
    ):
        assert isinstance(era5.profile_data_frame_workflow, mock.MagicMock)
        assert isinstance(era5.ozone_workflow, mock.MagicMock)
        assert isinstance(ancillary.write_dataframe, mock.MagicMock)
        assert not isinstance(ancillary.collect_era5_ancillary, mock.MagicMock)

        # simulate the workflow yielding dataframes
        m_data_frames = [mock.MagicMock(spec=pd.DataFrame) for _ in range(3)]
        m_profile_data_frame_workflow.return_value = iter(m_data_frames)

        ancillary.collect_era5_ancillary(acq, lat_longs, era5_data_dir, out_group)

        assert m_profile_data_frame_workflow.called
        assert m_write_dataframe.call_count == 3

        m_ozone_workflow.assert_called()
        assert m_write_scalar.call_count == 3
