from pathlib import Path
import os
from contextlib import ExitStack

from wagl.acquisition import acquisitions, Acquisition
from wagl.data import reproject_array_to_array
from wagl.geobox import GriddedGeoBox

import boto3
import rasterio
from osgeo import osr
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.transform import Affine
import numpy as np
import numpy.typing as npt

# TODO: tasks for integration
# * Change `Exception` to appropriate exception types for where-ever this gets moved to
# * 
# * 

COP30M_BUCKET_NAME = os.getenv("COP30M_BUCKET", "copernicus-dem-30m")
COP30M_CRS = osr.SpatialReference()
COP30M_CRS.ImportFromEPSG(4326)  # WGS84

test_cache = Path.home() / "cop30_cache"
test_cache.mkdir(parents=True, exist_ok=True)

def tile_rounding(value: float|int) -> int:
    """
    Rounds lat/lon coordinates into the Cop tile index.

    Cop DSM 30 and 90 datasets all use 1x1deg tiles, each assigned an latitude/longitude degree/index.

    All positive latitude/longitudes contain data for [x, x+1)
    All negative latitude/longitudes contain data for (x-1, x]
    The 0deg latitudes/longitude tiles contain data for 0..1

    TODO: Examples

    :param value:
        The value to be rounded into a tile coordinate.
    :return:
        The rounded tile index for the given coordinate.
    """
    if value >= 0.0:
        return int(value)
    else:
        return int(value)-1

def get_cop30m_for_extent(
    from_lat: int|float,
    to_lat: int|float,
    from_lon: int|float,
    to_lon: int|float,
    *,
    cop30_bucket: str = COP30M_BUCKET_NAME
) -> tuple[npt.NDArray[np.float32]|npt.NDArray[np.float64], Affine, osr.SpatialReference, float]:
    """
    Downloads Copernicus 30m DEM tiles that cover the lat/lon range requested.

    Non-integer ranges will be rounded away from zero, to ensure tiles that
    partially cover the required bounds are acquired.

    Input lat/lon coords are interpreted in the WGS 84 CRS.

    :param from_lat:
        The southern latitude for the region of the data to download.
    :param to_lat:
        The northern latitude for the region of the data to download.
    :param from_lon:
        The western longitude for the region of the data to download.
    :param to_lon:
        The eastern longitude for the region of the data to download.
    :param cop30_bucket:
        The S3 bucket name that hosts a copy of the COP 30 data to
        acquire DEM data from.
    """
    from_lat = tile_rounding(from_lat)
    to_lat = tile_rounding(to_lat)
    from_lon = tile_rounding(from_lon)
    to_lon = tile_rounding(to_lon)

    # Produce list of 'tiles' (1x1arcsecond regions the Cop 30 DEM is divided into)
    tiles: list[tuple[int, int]] = []
    for lat in range(from_lat, to_lat+1):
        for lon in range(from_lon, to_lon+1):
            tiles.append((lat, lon))

    datasets: list[MemoryFile] = []
    ds_crs: list[osr.SpatialReference] = []
    ds_res: list[tuple[int, int]] = []
    ds_nodata: list[tuple[int, int]] = []

    print("lat from:", from_lat, "to", to_lat)
    print("lon from:", from_lon, "to", to_lon)
    print("tiles:", tiles)

    s3 = boto3.client("s3")

    for (lat, lon) in tiles:
        lat_str = f"N{abs(lat):02d}" if lat >= 0 else f"S{abs(lat):02d}"
        lon_str = f"E{abs(lon):03d}" if lon >= 0 else f"W{abs(lon):03d}"
        key_id = f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM"
        key = f"{key_id}/{key_id}.tif"
        print(key_id)

        # TODO: extract to function
        try:
            # Download tile geotiff data directly into memory (~40MiB per tile)
            
            cache_path = test_cache / key
            if cache_path.exists():
                #print("Using cached", cache_path)
                buffer = MemoryFile(cache_path.read_bytes(), filename=os.path.basename(key))

            else:
                print("Downloading tile", key)
                buffer = MemoryFile(filename=os.path.basename(key))
                s3.download_fileobj(cop30_bucket, key, buffer)

                print("... caching.")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.touch()
                cache_path.write_bytes(buffer.getbuffer())

            datasets.append(buffer)

        except s3.exceptions.NoSuchKey:
            raise Exception(f"Failed to get cop30 DEM tile for {lat}, {lon}")
        
        with buffer.open() as ds:
            crs = osr.SpatialReference()
            crs.ImportFromWkt(ds.crs.to_wkt())

            ds_crs.append(crs)
            ds_res.append(ds.shape)
            ds_nodata.append(ds.nodata)

    # Sanity check all of the tiles are in the same CRS and resolution...
    #
    # NOTE: These shouldn't fail with the current cop DEM spec, but
    # just in case future versions allow for separate CRS across regional
    # areas... it's easy check and throw to catch possible future changes.
    # 
    # If these trigger in the future, work will need to be done to decide
    # what CRS/resolution to convert them all into first to minimise errors
    # during reprojection/resampling.
    if any(i.ExportToWkt() != COP30M_CRS.ExportToWkt() for i in ds_crs[1:]):
        raise Exception("Inconsistent CRS detected in DEM tiles")
    
    if any(i != ds_res[0] for i in ds_res[1:]):
        raise Exception("Inconsistent resolution detected in DEM tiles")
    
    if any(i != ds_nodata[0] for i in ds_nodata[1:]):
        raise Exception("Inconsistent nodata value detected in DEM tiles")

    # Mosaic the datasets
    #
    # NOTE: no extra arguments are needed if all the input datasets have
    # the same crs/resolution/nodata... it just does a basic painting
    # algorithm into an array with an extended resolution to fit all the data
    #
    # FIXME: somehow... this doesn't add all the tiles? (west bound column missing...)
    with ExitStack() as stack:
        src_tiles = [stack.enter_context(i.open()) for i in datasets]
        mosaic_data, mosaic_transform = merge(src_tiles)

    # Clean up tile buffer memory
    for ds in datasets:
        if not ds.closed:
            ds.close()

        del ds

    datasets.clear()

    return mosaic_data, mosaic_transform, ds_crs[0], ds_nodata[0]


def write_mosaic_tiff(
    filename: str,
    mosaic_data: npt.NDArray[np.float32]|npt.NDArray[np.float64],
    mosaic_transform: Affine,
    crs: osr.SpatialReference,
    nodata_value: float
):
    print("WRITING", filename, mosaic_data.shape, mosaic_data.dtype)
    if len(mosaic_data.shape) == 2:
        print("FROM shape", mosaic_data.shape)
        mosaic_data = mosaic_data.reshape(((1, mosaic_data.shape[0], mosaic_data.shape[1])))
        print("TO shape", mosaic_data.shape)

    # Create a mosaic dataset from the mosaic data
    mosaic_profile = {
        'driver': 'GTiff',
        'dtype': mosaic_data.dtype,
        'nodata': nodata_value,
        'width': mosaic_data.shape[-1],
        'height': mosaic_data.shape[-2],
        'count': 1,
        'crs': crs.ExportToWkt(),
        'transform': mosaic_transform,
        # Write a plain old untiled tiff
        'tiled': False,
        'compress': 'LZW'
    }

    with rasterio.open(filename, 'w', **mosaic_profile) as dst:
        dst.write(mosaic_data)


def _crs_transform(
    point: tuple[int|float, int|float],
    from_crs: osr.SpatialReference,
    to_crs: osr.SpatialReference
) -> tuple[float, float]:
    """
    A simple utility for transforming a point from one CRS into another
    without needing a GeoBox.

    :param point:
        The point to be trasnformed from `from_crs` to `to_crs`.
    :param from_crs:
        The spatial reference that `point` is in / being transformed from.
    :param to_crs:
        The spatial reference that `point` being transferred to.
    :return:
        The transformed point in the `to_crs` reference frame.
    """

    from_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    to_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(from_crs, to_crs)

    (x, y, _) = transform.TransformPoint(point[0], point[1])

    return (x, y)


def get_dem_for_acquisition(
    dataset: Acquisition,
    border_degrees: float
):
    """
    Get Copernicus DEM for a specified data acquisition.

    The returned data will be the mosaic of multiple DEM tiles, and transformed
    into the same CRS as the input acquisition the DEM was acquired for.

    :param dataset:
        The satellite acquisition to acquire the DEM data for.
    :param border_degrees:
        An optional buffer/border of data to acquire around the dataset's
        region of interest, in decimal degrees.
    :return:
        A tuple of (dem_data, dem_geobox) for the acquired DEM data.
    """
    # NOTE: In this function, using ds_ prefix for variables in dataset CRS coordinates
    # and border_ prefix for variables in WGS84 lat/lon coordinates.

    # Get the lat/lon extents of the acquisition (in degrees)
    ds_geobox = dataset.gridded_geo_box()
    border_extent = ds_geobox.project_extents(COP30M_CRS)
    border_ll: tuple[float, float] = (border_extent[0]-border_degrees, border_extent[1]-border_degrees)
    border_ur: tuple[float, float] = (border_extent[2]+border_degrees, border_extent[3]+border_degrees)
    ds_border = [*_crs_transform(border_ll, COP30M_CRS, ds_geobox.crs), *_crs_transform(border_ur, COP30M_CRS, ds_geobox.crs)]
    ds_ll = ds_geobox.ll
    ds_ur = ds_geobox.ur

    print(f"acquisition {dataset.band_name} ({dataset.band_id})")
    print("acquisition pixels", dataset.samples, dataset.lines)
    print("acquisition resolution", dataset.resolution)

    print("DEBUG CRS extent ll ur", ds_ll, ds_ur)
    print("DEBUG CRS border ll ur", ds_border)
    print("DEBUG WGS84 extent ll ur", border_extent)
    print("DEBUG WGS84 border ll ur", border_ll, border_ur)

    #print("extent width", ds_ur[0] - ds_ll[0])
    #print("extent degrees", border_extent[2] - border_extent[0])
    #print("bordered width", ds_border[2] - ds_border[0])
    #print("bordered degrees", border_ur[0] - border_ll[0])

    # Note: This is... not perfectly accurate (technically this is for the south-west)
    # - we are assuming equal border pixels on both sides...
    # (west, south, east, north)
    ds_border_size = (
        (ds_ll[0] - ds_border[0]),
        (ds_ll[1] - ds_border[1]),
        (ds_border[2] - ds_ur[0]),
        (ds_border[3] - ds_ur[1])
    )

    # TBD: not sure what the preferred rounding is here, probably doesn't matter..?
    ds_border_size_px = (
        int(ds_border_size[0] / dataset.resolution[0]), int(ds_border_size[1] / dataset.resolution[1]),
        int(ds_border_size[2] / dataset.resolution[0]), int(ds_border_size[3] / dataset.resolution[1])
    )

    ds_border_size_deg = (
        border_extent[0] - border_ll[0],
        border_extent[1] - border_ll[1],
        (border_ur[0] - border_extent[2]),
        (border_ur[1] - border_extent[3])
    )

    #print("(west, south, east, north)")
    #print("ds_border_size:", ds_border_size)
    #print("ds_border_size_px:", ds_border_size_px)
    #print("ds_border_size_deg:", ds_border_size_deg)

    # Get DEM for acquisition
    extent = (border_ll[1], border_ur[1], border_ll[0], border_ur[0])
    dem_data, dem_transform, dem_crs, dem_nodata = get_cop30m_for_extent(*extent)

    print("==========")
    print("==========")
    print("dem shape:", dem_data.shape)
    print("dem dtype:", dem_data.dtype)
    print("dem nodata:", dem_nodata)
    #print("dem crs:", dem_crs)
    #print("dem transform:", dem_transform)

    write_mosaic_tiff(
        str(test_cache / "test_mosaic_{border_degrees}deg_border.tif"),
        dem_data, dem_transform, dem_crs, dem_nodata
    )

    # Create geobox for DEM
    dem_origin = dem_transform * (0,0)
    print("dem origin:", dem_origin)
    dem_origin_ur = dem_transform * (1,1)
    dem_pixelsize = (dem_origin_ur[0] - dem_origin[0], dem_origin[1] - dem_origin_ur[1])
    print("dem pixelsize:", dem_pixelsize)
    dem_geobox = GriddedGeoBox(
        shape=dem_data.shape,
        origin=dem_origin,
        pixelsize=dem_pixelsize,
        crs=dem_crs
    )

    print("test CRS dem pixel size", _crs_transform(dem_origin_ur, COP30M_CRS, ds_geobox.crs)[0] - _crs_transform(dem_origin, COP30M_CRS, ds_geobox.crs)[0])

    # Reproject DEM into same CRS and pixel size as acquisition
    new_dem_shape = (
        ds_geobox.x_size() + ds_border_size_px[0] + ds_border_size_px[2],
        ds_geobox.y_size() + ds_border_size_px[1] + ds_border_size_px[3]
    )

    print("reprojecting...")
    print("new_dem_shape:", new_dem_shape)
    
    new_dem_geobox = GriddedGeoBox(
        shape=new_dem_shape,
        origin=ds_geobox.convert_coordinates((-ds_border_size_px[0], -ds_border_size_px[1])),
        pixelsize=ds_geobox.pixelsize,
        crs=ds_geobox.crs.ExportToWkt()
    )

    new_dem_data = reproject_array_to_array(
        dem_data,
        dem_geobox,
        new_dem_geobox,
        src_nodata=dem_nodata,
        dst_nodata=dem_nodata,
        resampling=Resampling.bilinear
    )

    write_mosaic_tiff(
        str(test_cache / "test_reproj_{border_degrees}deg_border.tif"),
        new_dem_data.reshape((1, new_dem_shape[0], new_dem_shape[1])),
        new_dem_geobox.transform, new_dem_geobox.crs, dem_nodata
    )

    return new_dem_data, new_dem_geobox

def test():
    #scene_path = "/usr/src/wagl/LC80400332013190LGN03"
    scene_path = "/usr/src/wagl/LC08_L1TP_028030_20221018_20221031_02_T1"

    acqs = acquisitions(scene_path)
    band = acqs.get_all_acquisitions()[0]

    get_dem_for_acquisition(band, 1.0)

if __name__ == "__main__":
    test()
