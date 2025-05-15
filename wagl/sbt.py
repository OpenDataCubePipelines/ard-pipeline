"""
SBT: Surface Brightness Temperature Ancillaries.

This code was formerly in the ancillary module. The SBT feature's code size &
complexity complicated the ancillary module. Additionally, as SBT was disabled,
it is essentially dead code.

This module temporarily retains SBT code *outside* the ancillary module, separate
the different concerns of these modules. This follows the ERA5 & MERRA2 approach
of having a separate module for ancillary specialisation.
"""

from posixpath import join as ppjoin

import pandas as pd

from wagl.ancillary import ECWMF_LEVELS, AncillaryError
from wagl.atmos import kelvin_2_celcius, relative_humdity
from wagl.constants import (
    GEOPOTENTIAL_HEIGHT,
    POINT_FMT,
    PRESSURE,
    RELATIVE_HUMIDITY,
    TEMPERATURE,
    DatasetName,
)
from wagl.hdf5 import (
    H5CompressionFilter,
    write_dataframe,
    write_scalar,
)


def collect_sbt_ancillary(
    acquisition,
    lonlats,
    ancillary_path,
    invariant_fname=None,
    out_group=None,
    compression=H5CompressionFilter.LZF,
    filter_opts=None,
):
    """Collects the ancillary data required for surface brightness
    temperature.

    :param acquisition:
        An instance of an `Acquisition` object.

    :param lonlats:
        A `list` of tuples containing (longitude, latitude) coordinates.

    :param ancillary_path:
        A `str` containing the directory pathname to the ECMWF
        ancillary data.

    :param invariant_fname:
        A `str` containing the file pathname to the invariant geopotential
        data.

    :param out_group:
        A writeable HDF5 `Group` object.

    :param compression:
        The compression filter to use.
        Default is H5CompressionFilter.LZF

    :param filter_opts:
        A dict of key value pairs available to the given configuration
        instance of H5CompressionFilter. For example
        H5CompressionFilter.LZF has the keywords *chunks* and *shuffle*
        available.
        Default is None, which will use the default settings for the
        chosen H5CompressionFilter instance.

    :return:
        An opened `h5py.File` object, that is either in-memory using the
        `core` driver, or on disk.
    """
    assert out_group is not None
    fid = out_group

    fid.attrs["sbt-ancillary"] = True

    dt = acquisition.acquisition_datetime

    description = (
        "Combined Surface and Pressure Layer data retrieved from "
        "the ECWMF catalogue."
    )
    attrs = {"description": description, "Date used for querying ECWMF": dt}

    for i, lonlat in enumerate(lonlats):
        pnt = POINT_FMT.format(p=i)
        # get data located at the surface
        dew = ecwmf_dewpoint_temperature(ancillary_path, lonlat, dt)
        t2m = ecwmf_temperature_2metre(ancillary_path, lonlat, dt)
        sfc_prs = ecwmf_surface_pressure(ancillary_path, lonlat, dt)
        sfc_hgt = ecwmf_elevation(invariant_fname, lonlat)
        sfc_rh = relative_humdity(t2m[0], dew[0])

        # output the scalar data along with the attrs
        dname = ppjoin(pnt, DatasetName.DEWPOINT_TEMPERATURE.value)
        write_scalar(dew[0], dname, fid, dew[1])

        dname = ppjoin(pnt, DatasetName.TEMPERATURE_2M.value)
        write_scalar(t2m[0], dname, fid, t2m[1])

        dname = ppjoin(pnt, DatasetName.SURFACE_PRESSURE.value)
        write_scalar(sfc_prs[0], dname, fid, sfc_prs[1])

        dname = ppjoin(pnt, DatasetName.SURFACE_GEOPOTENTIAL.value)
        write_scalar(sfc_hgt[0], dname, fid, sfc_hgt[1])

        dname = ppjoin(pnt, DatasetName.SURFACE_RELATIVE_HUMIDITY.value)
        attrs = {"description": "Relative Humidity calculated at the surface"}
        write_scalar(sfc_rh, dname, fid, attrs)

        # get the data from each of the pressure levels (1 -> 1000 ISBL)
        gph = ecwmf_geo_potential(ancillary_path, lonlat, dt)
        tmp = ecwmf_temperature(ancillary_path, lonlat, dt)
        rh = ecwmf_relative_humidity(ancillary_path, lonlat, dt)

        dname = ppjoin(pnt, DatasetName.GEOPOTENTIAL.value)
        write_dataframe(
            gph[0], dname, fid, compression, attrs=gph[1], filter_opts=filter_opts
        )

        dname = ppjoin(pnt, DatasetName.TEMPERATURE.value)
        write_dataframe(
            tmp[0], dname, fid, compression, attrs=tmp[1], filter_opts=filter_opts
        )

        dname = ppjoin(pnt, DatasetName.RELATIVE_HUMIDITY.value)
        write_dataframe(
            rh[0], dname, fid, compression, attrs=rh[1], filter_opts=filter_opts
        )

        # combine the surface and higher pressure layers into a single array
        cols = [GEOPOTENTIAL_HEIGHT, PRESSURE, TEMPERATURE, RELATIVE_HUMIDITY]
        layers = pd.DataFrame(
            columns=cols, index=range(rh[0].shape[0]), dtype="float64"
        )

        layers[GEOPOTENTIAL_HEIGHT] = gph[0][GEOPOTENTIAL_HEIGHT].values
        layers[PRESSURE] = ECWMF_LEVELS[::-1]
        layers[TEMPERATURE] = tmp[0][TEMPERATURE].values
        layers[RELATIVE_HUMIDITY] = rh[0][RELATIVE_HUMIDITY].values

        # define the surface level
        df = pd.DataFrame(
            {
                GEOPOTENTIAL_HEIGHT: sfc_hgt[0],
                PRESSURE: sfc_prs[0],
                TEMPERATURE: kelvin_2_celcius(t2m[0]),
                RELATIVE_HUMIDITY: sfc_rh,
            },
            index=[0],
        )

        # MODTRAN requires the height to be ascending
        # and the pressure to be descending
        wh = (layers[GEOPOTENTIAL_HEIGHT] > sfc_hgt[0]) & (
            layers[PRESSURE] < sfc_prs[0].round()
        )
        df = df.append(layers[wh])
        df.reset_index(drop=True, inplace=True)

        dname = ppjoin(pnt, DatasetName.ATMOSPHERIC_PROFILE.value)
        write_dataframe(
            df, dname, fid, compression, attrs=attrs, filter_opts=filter_opts
        )

        fid[pnt].attrs["lonlat"] = lonlat


def ecwmf_elevation(datafile, lonlat):
    """Retrieve a pixel from the ECWMF invariant geo-potential
    dataset.
    Converts to Geo-Potential height in KM.
    2 metres is added to the result before returning.
    """
    # TODO; have swfo convert the files to HDF5
    raise AncillaryError("No Invariant Geo-Potential data")
    # try:
    #     data = get_pixel(datafile, lonlat) / 9.80665 / 1000.0 + 0.002
    # except IndexError:
    #     raise AncillaryError("No Invariant Geo-Potential data")

    # url = urlparse(datafile, scheme='file').geturl()

    # metadata = {'data_source': 'ECWMF Invariant Geo-Potential',
    #             'url': url}

    # # ancillary metadata tracking
    # md = extract_ancillary_metadata(datafile)
    # for key in md:
    #     metadata[key] = md[key]

    # return data, metadata


def ecwmf_temperature_2metre(input_path, lonlat, time):
    """Retrieve a pixel value from the ECWMF 2 metre Temperature
    collection.
    """
    # TODO; have swfo convert the files to HDF5
    raise AncillaryError("No ECWMF 2 metre Temperature data")
    # product = DatasetName.TEMPERATURE_2M.value.lower()
    # search = pjoin(input_path, DatasetName.ECMWF_PATH_FMT.value)
    # files = glob.glob(search.format(product=product, year=time.year))
    # data = None
    # required_ymd = datetime.datetime(time.year, time.month, time.day)
    # for f in files:
    #     url = urlparse(f, scheme='file').geturl()
    #     ymd = splitext(basename(f))[0].split('_')[1]
    #     ancillary_ymd = datetime.datetime.strptime(ymd, '%Y-%m-%d')
    #     if ancillary_ymd == required_ymd:
    #         data = get_pixel(f, lonlat)

    #         metadata = {'data_source': 'ECWMF 2 metre Temperature',
    #                     'url': url,
    #                     'query_date': time}

    #         # ancillary metadata tracking
    #         md = extract_ancillary_metadata(f)
    #         for key in md:
    #             metadata[key] = md[key]

    #         return data, metadata

    # if data is None:
    #     raise AncillaryError("No ECWMF 2 metre Temperature data")


def ecwmf_dewpoint_temperature(input_path, lonlat, time):
    """Retrieve a pixel value from the ECWMF 2 metre Dewpoint
    Temperature collection.
    """
    # TODO; have swfo convert the files to HDF5
    raise AncillaryError("No ECWMF 2 metre Dewpoint Temperature data")
    # product = DatasetName.DEWPOINT_TEMPERATURE.value.lower()
    # search = pjoin(input_path, DatasetName.ECMWF_PATH_FMT.value)
    # files = glob.glob(search.format(product=product, year=time.year))
    # data = None
    # required_ymd = datetime.datetime(time.year, time.month, time.day)
    # for f in files:
    #     url = urlparse(f, scheme='file').geturl()
    #     ymd = splitext(basename(f))[0].split('_')[1]
    #     ancillary_ymd = datetime.datetime.strptime(ymd, '%Y-%m-%d')
    #     if ancillary_ymd == required_ymd:
    #         data = get_pixel(f, lonlat)

    #         metadata = {'data_source': 'ECWMF 2 metre Dewpoint Temperature ',
    #                     'url': url,
    #                     'query_date': time}

    #         # ancillary metadata tracking
    #         md = extract_ancillary_metadata(f)
    #         for key in md:
    #             metadata[key] = md[key]

    #         return data, metadata

    # if data is None:
    #     raise AncillaryError("No ECWMF 2 metre Dewpoint Temperature data")


def ecwmf_surface_pressure(input_path, lonlat, time):
    """Retrieve a pixel value from the ECWMF Surface Pressure
    collection.
    Scales the result by 100 before returning.
    """
    # TODO; have swfo convert the files to HDF5
    raise AncillaryError("No ECWMF Surface Pressure data")
    # product = DatasetName.SURFACE_PRESSURE.value.lower()
    # search = pjoin(input_path, DatasetName.ECMWF_PATH_FMT.value)
    # files = glob.glob(search.format(product=product, year=time.year))
    # data = None
    # required_ymd = datetime.datetime(time.year, time.month, time.day)
    # for f in files:
    #     url = urlparse(f, scheme='file').geturl()
    #     ymd = splitext(basename(f))[0].split('_')[1]
    #     ancillary_ymd = datetime.datetime.strptime(ymd, '%Y-%m-%d')
    #     if ancillary_ymd == required_ymd:
    #         data = get_pixel(f, lonlat) / 100.0

    #         metadata = {'data_source': 'ECWMF Surface Pressure',
    #                     'url': url,
    #                     'query_date': time}

    #         # ancillary metadata tracking
    #         md = extract_ancillary_metadata(f)
    #         for key in md:
    #             metadata[key] = md[key]

    #         return data, metadata

    # if data is None:
    #     raise AncillaryError("No ECWMF Surface Pressure data")


def ecwmf_water_vapour(input_path, lonlat, time):
    """Retrieve a pixel value from the ECWMF Total Column Water Vapour
    collection.
    """
    # TODO; have swfo convert the files to HDF5
    raise AncillaryError("No ECWMF Total Column Water Vapour data")
    # product = DatasetName.WATER_VAPOUR.value.lower()
    # search = pjoin(input_path, DatasetName.ECMWF_PATH_FMT.value)
    # files = glob.glob(search.format(product=product, year=time.year))
    # data = None
    # required_ymd = datetime.datetime(time.year, time.month, time.day)
    # for f in files:
    #     url = urlparse(f, scheme='file').geturl()
    #     ymd = splitext(basename(f))[0].split('_')[1]
    #     ancillary_ymd = datetime.datetime.strptime(ymd, '%Y-%m-%d')
    #     if ancillary_ymd == required_ymd:
    #         data = get_pixel(f, lonlat)

    #         metadata = {'data_source': 'ECWMF Total Column Water Vapour',
    #                     'url': url,
    #                     'query_date': time}

    #         # ancillary metadata tracking
    #         md = extract_ancillary_metadata(f)
    #         for key in md:
    #             metadata[key] = md[key]

    #         return data, metadata

    # if data is None:
    #     raise AncillaryError("No ECWMF Total Column Water Vapour data")


def ecwmf_temperature(input_path, lonlat, time):
    """Retrieve a pixel value from the ECWMF Temperature collection
    across 37 height pressure levels, for a given longitude,
    latitude and time.

    Reverses the order of elements
    (1000 -> 1 mb, rather than 1 -> 1000 mb) before returning.
    """
    # TODO; have swfo convert the files to HDF5
    raise AncillaryError("No ECWMF Temperature profile data")
    # product = DatasetName.TEMPERATURE.value.lower()
    # search = pjoin(input_path, DatasetName.ECMWF_PATH_FMT.value)
    # files = glob.glob(search.format(product=product, year=time.year))
    # data = None
    # required_ymd = datetime.datetime(time.year, time.month, time.day)
    # for f in files:
    #     url = urlparse(f, scheme='file').geturl()
    #     ymd = splitext(basename(f))[0].split('_')[1]
    #     ancillary_ymd = datetime.datetime.strptime(ymd, '%Y-%m-%d')
    #     if ancillary_ymd == required_ymd:
    #         bands = list(range(1, 38))
    #         data = get_pixel(f, lonlat, bands)[::-1]

    #         metadata = {'data_source': 'ECWMF Temperature',
    #                     'url': url,
    #                     'query_date': time}

    #         # ancillary metadata tracking
    #         md = extract_ancillary_metadata(f)
    #         for key in md:
    #             metadata[key] = md[key]

    #         # internal file metadata (and reverse the ordering)
    #         df = read_metadata_tags(f, bands).iloc[::-1]
    #         df.insert(0, 'Temperature', data)

    #         return df, metadata

    # if data is None:
    #     raise AncillaryError("No ECWMF Temperature profile data")


def ecwmf_geo_potential(input_path, lonlat, time):
    """Retrieve a pixel value from the ECWMF Geo-Potential collection
    across 37 height pressure levels, for a given longitude,
    latitude and time.

    Converts to geo-potential height in KM, and reverses the order of
    the elements (1000 -> 1 mb, rather than 1 -> 1000 mb) before
    returning.
    """
    # TODO; have swfo convert the files to HDF5
    raise AncillaryError("No ECWMF Geo-Potential profile data")
    # product = DatasetName.GEOPOTENTIAL.value.lower()
    # search = pjoin(input_path, DatasetName.ECMWF_PATH_FMT.value)
    # files = glob.glob(search.format(product=product, year=time.year))
    # data = None
    # required_ymd = datetime.datetime(time.year, time.month, time.day)
    # for f in files:
    #     url = urlparse(f, scheme='file').geturl()
    #     ymd = splitext(basename(f))[0].split('_')[1]
    #     ancillary_ymd = datetime.datetime.strptime(ymd, '%Y-%m-%d')
    #     if ancillary_ymd == required_ymd:
    #         bands = list(range(1, 38))
    #         data = get_pixel(f, lonlat, bands)[::-1]
    #         scaled_data = data / 9.80665 / 1000.0

    #         metadata = {'data_source': 'ECWMF Geo-Potential',
    #                     'url': url,
    #                     'query_date': time}

    #         # ancillary metadata tracking
    #         md = extract_ancillary_metadata(f)
    #         for key in md:
    #             metadata[key] = md[key]

    #         # internal file metadata (and reverse the ordering)
    #         df = read_metadata_tags(f, bands).iloc[::-1]
    #         df.insert(0, 'GeoPotential', data)
    #         df.insert(1, 'GeoPotential_Height', scaled_data)

    #         return df, md

    # if data is None:
    #     raise AncillaryError("No ECWMF Geo-Potential profile data")


def ecwmf_relative_humidity(input_path, lonlat, time):
    """Retrieve a pixel value from the ECWMF Relative Humidity collection
    across 37 height pressure levels, for a given longitude,
    latitude and time.

    Reverses the order of elements
    (1000 -> 1 mb, rather than 1 -> 1000 mb) before returning.
    """
    # TODO; have swfo convert the files to HDF5
    raise AncillaryError("No ECWMF Relative Humidity profile data")
    # product = DatasetName.RELATIVE_HUMIDITY.value.lower()
    # search = pjoin(input_path, DatasetName.ECMWF_PATH_FMT.value)
    # files = glob.glob(search.format(product=product, year=time.year))
    # data = None
    # required_ymd = datetime.datetime(time.year, time.month, time.day)
    # for f in files:
    #     url = urlparse(f, scheme='file').geturl()
    #     ymd = splitext(basename(f))[0].split('_')[1]
    #     ancillary_ymd = datetime.datetime.strptime(ymd, '%Y-%m-%d')
    #     if ancillary_ymd == required_ymd:
    #         bands = list(range(1, 38))
    #         data = get_pixel(f, lonlat, bands)[::-1]

    #         metadata = {'data_source': 'ECWMF Relative Humidity',
    #                     'url': url,
    #                     'query_date': time}

    #         # file level metadata
    #         md = extract_ancillary_metadata(f)
    #         for key in md:
    #             metadata[key] = md[key]

    #         # internal file metadata (and reverse the ordering)
    #         df = read_metadata_tags(f, bands).iloc[::-1]
    #         df.insert(0, 'Relative_Humidity', data)

    #         return df, metadata

    # if data is None:
    #     raise AncillaryError("No ECWMF Relative Humidity profile data")
