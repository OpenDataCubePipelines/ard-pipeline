from enum import Enum, auto
import datetime as dt
from pathlib import Path
import zipfile

# Note: while this is the official CDS API... their own forums & the git repo indicate this is
# potentially 
import cdsapi
from osgeo import osr
import rasterio
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from rasterio.transform import Affine

import numpy as np
import numpy.typing as npt

from wagl.acquisition import acquisitions, Acquisition
from wagl.data import reproject_array_to_array
from wagl.geobox import GriddedGeoBox
from wagl.acquisition.copernicus_dem import write_mosaic_tiff, _crs_transform

test_cache = Path.home() / "ecmwf_cache"
test_cache.mkdir(parents=True, exist_ok=True)


# TODO: for integration into WAGL... CDS/ADS credentials to be loaded from somewhere..
CDS_ERA5_URL = "https://cds.climate.copernicus.eu/api"
CDS_ERA5_KEY = "<CDS API key goes here>"

ADS_CAMS_URL = "https://ads.atmosphere.copernicus.eu/api"
ADS_CAMS_KEY = "<CDS API key goes here>"

# ERA5 NetCDF data is effectively WGS84
# Reference: https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-SpatialgridSpatialGrid
#
# "ERA5 data is referenced in the horizontal with respect to the WGS84 ellipse (which defines the major/minor axes)",
# where "horizontal" in this context refers to lat/lon coords (and "vertical" would refer to altitude).
ECMWF_CRS = osr.SpatialReference()
ECMWF_CRS.ImportFromEPSG(4326)  # WGS84


class ECMWFProduct(Enum):
    ERA5_OZONE = auto()
    ERA5_WATER_VAPOUR = auto()
    CAMS_GLOBAL_FORECAST_TAOD_550nm = auto()


def get_ecmwf_params_for_product_extent(
    product: ECMWFProduct,
    timestamp: dt.datetime,
    from_lat: int|float,
    to_lat: int|float,
    from_lon: int|float,
    to_lon: int|float,
) -> tuple[str, dict[str, object]]:
    if product == ECMWFProduct.ERA5_OZONE or product == ECMWFProduct.ERA5_WATER_VAPOUR:
        dataset = "reanalysis-era5-single-levels"

        if product == ECMWFProduct.ERA5_OZONE:
            product_var_name = 'total_column_ozone'
        elif product == ECMWFProduct.ERA5_WATER_VAPOUR:
            product_var_name = 'total_column_water_vapour'
        else:
            raise Exception("Usupported product")

        request = {
            'product_type': ['reanalysis'],
            'variable': [product_var_name],
            'year': [str(timestamp.year)], # eg: ['2023'],
            'month': [f"{timestamp.month:02d}"], # eg: ['04'],
            'day': [f"{timestamp.day:02d}"], # eg: ['19'],
            'time': [timestamp.strftime("%H:00")], # eg: ['08:00'],
            'data_format': 'netcdf',
            'download_format': 'unarchived',
            # [north, west, south, east]
            'area': [to_lat, from_lon, from_lat, to_lon], # eg: [-30.74, 137.71, -42.07, 153.57]
        }

        return dataset, request

    elif product == ECMWFProduct.CAMS_GLOBAL_FORECAST_TAOD_550nm:
        dataset = "cams-global-atmospheric-composition-forecasts"
        date_str = timestamp.strftime("%Y-%m-%d")
        # CAMS TOAD only has 00:00 and 12:00 forecasts
        time_str = "00:00" if timestamp.hour < 12 else "12:00"

        request = {
            'variable': ['total_aerosol_optical_depth_550nm'],
            'date': [f"{date_str}/{date_str}"], # eg: ['2024-09-22/2024-09-22'],
            'time': [time_str], # eg: ['12:00'],
            'leadtime_hour': ['0'],
            'type': ['forecast'],
            'data_format': 'netcdf_zip', # eg: 'netcdf_zip',
            'area': [to_lat, from_lon, from_lat, to_lon] # eg: [-29.13, 139.33, -40.53, 153.29]
        }

        return dataset, request

    else:
        raise Exception("Unsupported WCMWF product")


def get_ecmwf_for_extent(
    product: ECMWFProduct,
    timestamp: dt.datetime,
    from_lat: int|float,
    to_lat: int|float,
    from_lon: int|float,
    to_lon: int|float
) -> tuple[npt.NDArray[np.float32]|npt.NDArray[np.float64], Affine, osr.SpatialReference, float]:

    dataset, request = get_ecmwf_params_for_product_extent(
        product, timestamp,
        from_lat, to_lat, from_lon, to_lon
    )

    print(f"Downloading {product.name} for {request['area']}")

    # TODO: for integration into wagl... this funtion will need a path to a scratch dir to write files to
    # - because the CDS API unfortunately only allows downloading to a file (not into an in-memory stream/buffer)
    filename = f'temp_{product.name}_{from_lat}_{to_lat}_to_{from_lon}_{to_lon}.nc'
    cache_path = test_cache / filename
    if cache_path.exists():
        print("Using cached", cache_path)

    else:
        # NOTE: By default, the CDS API client will wait until a request is completed
        # before running further functions on the request (eg: .download()).
        #
        # This means by default, this client can .retrieve().download() and the API itself
        # takes care of waiting for the request before attempting the download.
        if product == ECMWFProduct.CAMS_GLOBAL_FORECAST_TAOD_550nm:
            client = cdsapi.Client(url=ADS_CAMS_URL, key=ADS_CAMS_KEY)
        else:
            client = cdsapi.Client(url=CDS_ERA5_URL, key=CDS_ERA5_KEY)

        # FIXME: ECMWF's API doesn't support anything besides downloading to physical files
        # - so we can't download into memory with this approach (their HTTP API is trivial though,
        # - could do what the API does and just do the HTTP GET ourselves...)
        client.retrieve(dataset, request).download(str(cache_path))

        # Handle zip files (which only ever contain a single .nc file)
        if req.location.endswith(".zip"):
            zip_path = cache_path.rename(cache_path.with_suffix(".nc.zip"))

            with zip_path.open("rb") as zf:
                zip = zipfile.ZipFile(zf)
                filenames = zip.namelist()
                if len(filenames) != 1:
                    raise Exception("Unexpected zip file, found more than one file in archive!")
                
                cache_path.write_bytes(zip.read(filenames[0]))

    # Read and return the NetCDF data (just a simple single band raster image)
    with rasterio.open(cache_path) as ds:
        data = ds.read(1)
        # Note: their NetCDF files shouldn't have a CRS in any standard form,
        # their NetCDF is just a package of resampled GRIB data to look like a
        # raster format - and even the GRIB metadata does not specify a CRS of
        # any kind, the GRIB data spec 'does' specify the spatial reference for
        # all data though.  See ECMWF_CRS comment for reference.
        crs = ECMWF_CRS

        # Apply data tarnsformations from ERA5 units into MODTRAN units
        if product == ECMWFProduct.ERA5_OZONE:
            # From Fuqin:
            #    unit: kg m^-2. Need to convert. The conversion is: 1 DU = 2.1415E-5 kg m^-2 
            #    Modtran uses ATM-CM, 1 ATM-CM=1000 DU (dobson)
            #    Data range is usually 0.2-0.3 ATM-CM

            print("ozone min/max/mean BEFORE (kg m^-2)", data.min(), data.max(), data.mean())
            divisor = 2.1415E-5 * 1000 # (kg m^-2 -> DU -> ATM-CM)
            data /= divisor
            print("ozone min/max/mean AFTER (ATM-CM)", data.min(), data.max(), data.mean())

        elif product == ECMWFProduct.ERA5_WATER_VAPOUR:
            # From Fuqin:
            #    unit: kg m^-2 
            #    MODTRAN uses g / cm^2.  1 kg m^-2 = 10 g / cm^2
            #    Therefore the data need to divide 10 to convert to g / cm^2
            #    Data range is usually 0-5 g / cm^2

            print("water vapour min/max/mean BEFORE (kg m^-2)", data.min(), data.max(), data.mean())
            data /= 10
            print("water vapour min/max/mean AFTER (g m^-2)", data.min(), data.max(), data.mean())

        elif product == ECMWFProduct.CAMS_GLOBAL_FORECAST_TAOD_550nm:
            # Nothing to do... already in desired units
            pass

        return data, ds.transform, crs, ds.nodata

def get_ecmwf_for_acquisition(
    dataset: Acquisition,
    product: ECMWFProduct,
    border_degrees: float = 0.0
):
    """
    Get ECMWF products for a specified data acquisition.

    The returned data will most likely NOT be in the dataset's spatial reference,
    this is intentional - the ECMWF ancillary data is spatially very low
    resolution and as such it's better to sample this data directly than reproject
    and resample it into the acquisition's CRS and/or resolution.
    """
    # NOTE: In this function, using ds_ prefix for variables in dataset CRS coordinates
    # and border_ prefix for variables in WGS84 lat/lon coordinates.

    # Get the lat/lon extents of the acquisition (in degrees)
    ds_geobox = dataset.gridded_geo_box()
    border_extent = ds_geobox.project_extents(ECMWF_CRS)
    border_ll: tuple[float, float] = (border_extent[0]-border_degrees, border_extent[1]-border_degrees)
    border_ur: tuple[float, float] = (border_extent[2]+border_degrees, border_extent[3]+border_degrees)

    print(f"acquisition {dataset.band_name} ({dataset.band_id})")
    print("acquisition pixels", dataset.samples, dataset.lines)
    print("acquisition resolution", dataset.resolution)

    # Get aux data for acquisition
    extent = (border_ll[1], border_ur[1], border_ll[0], border_ur[0])
    aux_data, aux_transform, aux_crs, aux_nodata = get_ecmwf_for_extent(
        product, dataset.acquisition_datetime,
        *extent
    )

    # Create geobox for DEM
    aux_origin = aux_transform * (0,0)
    aux_origin_ur = aux_transform * (1,1)
    aux_pixelsize = (aux_origin_ur[0] - aux_origin[0], aux_origin[1] - aux_origin_ur[1])
    aux_geobox = GriddedGeoBox(
        shape=aux_data.shape,
        origin=aux_origin,
        pixelsize=aux_pixelsize,
        crs=aux_crs
    )

    return aux_data, aux_nodata, aux_geobox

def test():
    scene_path = "/usr/src/wagl/LC08_L1TP_028030_20221018_20221031_02_T1"

    acqs = acquisitions(scene_path)
    band = acqs.get_all_acquisitions()[0]

    ozone, ozone_nodata, ozone_geobox = get_ecmwf_for_acquisition(band, ECMWFProduct.ERA5_OZONE, 1.0)
    print("ozone origin:", ozone_geobox.ul)
    print("ozone pixelsize:", ozone_geobox.pixelsize)
    print("ozone mean", ozone.mean())
    write_mosaic_tiff(
        "ozone.tif",
        ozone, ozone_geobox.transform, ozone_geobox.crs, ozone_nodata
    )

    water_vapour, water_vapour_nodata, water_vapour_geobox = get_ecmwf_for_acquisition(band, ECMWFProduct.ERA5_WATER_VAPOUR, 1.0)
    print("water vapour origin:", water_vapour_geobox.ul)
    print("water vapour pixelsize:", water_vapour_geobox.pixelsize)
    print("water vapour mean", water_vapour.mean())
    write_mosaic_tiff(
        "water_vapour.tif",
        water_vapour, water_vapour_geobox.transform, water_vapour_geobox.crs, water_vapour_nodata
    )

    total_aerosol, total_aerosol_nodata, total_aerosol_geobox = get_ecmwf_for_acquisition(band, ECMWFProduct.CAMS_GLOBAL_FORECAST_TAOD_550nm, 1.0)
    print("total aerosol origin:", total_aerosol_geobox.ul)
    print("total aerosol pixelsize:", total_aerosol_geobox.pixelsize)
    print("total aerosol mean", total_aerosol.mean())
    write_mosaic_tiff(
        "total_aerosol.tif",
        total_aerosol, total_aerosol_geobox.transform, total_aerosol_geobox.crs, total_aerosol_nodata
    )

    # TODO: sample with grid, produce mean, etc - like wagl does

if __name__ == "__main__":
    test()
