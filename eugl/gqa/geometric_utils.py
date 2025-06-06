#!/usr/bin/env python
#
# Functions in this file represent work to be refactored.

import datetime
import logging
import typing

import numpy as np
import rasterio
import yaml
from rasterio.warp import Resampling

from eugl.fmask import run_command

# Post SLC-OFF date
SLC_OFF = datetime.datetime(2003, 6, 1)
_LOG = logging.getLogger(__name__)

# TODO only work with the latest naming convention provided in the MTL file
# TODO replace the quick and dirty BAND_MAP that accounts for different sensors
#      different naming conventions (even of the same satellite)
BAND_MAP = {
    "LE7": {
        "LS5": {
            "1": "B1",
            "2": "B2",
            "3": "B3",
            "4": "B4",
            "5": "B5",
            "6": "B6_VCID_1",
            "7": "B7",
        },
        "LS7": {
            "1": "B1",
            "2": "B2",
            "3": "B3",
            "4": "B4",
            "5": "B5",
            "61": "B6_VCID_1",
            "62": "B6_VCID_2",
            "7": "B7",
        },
        "LS8": {
            "1": "B1",
            "2": "B1",
            "3": "B2",
            "4": "B3",
            "5": "B4",
            "6": "B5",
            "7": "B7",
            "10": "B6_VCID_1",
            "11": "B6_VCID_1",
        },
        "LS9": {
            "1": "B1",
            "2": "B1",
            "3": "B2",
            "4": "B3",
            "5": "B4",
            "6": "B5",
            "7": "B7",
            "10": "B6_VCID_1",
            "11": "B6_VCID_1",
        },
        "S2A": {
            "1": "B1",
            "2": "B1",
            "3": "B2",
            "4": "B3",
            "5": "B3",
            "6": "B3",
            "7": "B3",
            "8": "B4",
            "8A": "B4",
            "11": "B5",
            "12": "B7",
        },
        "S2B": {
            "1": "B1",
            "2": "B1",
            "3": "B2",
            "4": "B3",
            "5": "B3",
            "6": "B3",
            "7": "B3",
            "8": "B4",
            "8A": "B4",
            "11": "B5",
            "12": "B7",
        },
        "S2C": {
            "1": "B1",
            "2": "B1",
            "3": "B2",
            "4": "B3",
            "5": "B3",
            "6": "B3",
            "7": "B3",
            "8": "B4",
            "8A": "B4",
            "11": "B5",
            "12": "B7",
        },
    },
    "LT5": {
        "LS5": {
            "1": "B1",
            "2": "B2",
            "3": "B3",
            "4": "B4",
            "5": "B5",
            "6": "B6",
            "7": "B7",
        },
        "LS7": {
            "1": "B1",
            "2": "B2",
            "3": "B3",
            "4": "B4",
            "5": "B5",
            "61": "B6_VCID_1",
            "62": "B6_VCID_2",
            "7": "B7",
        },
        "LS8": {
            "1": "B1",
            "2": "B1",
            "3": "B2",
            "4": "B3",
            "5": "B4",
            "6": "B5",
            "7": "B7",
            "10": "B6_VCID_1",
            "11": "B6_VCID_1",
        },
        "LS9": {
            "1": "B1",
            "2": "B1",
            "3": "B2",
            "4": "B3",
            "5": "B4",
            "6": "B5",
            "7": "B7",
            "10": "B6_VCID_1",
            "11": "B6_VCID_1",
        },
        "S2A": {
            "1": "B1",
            "2": "B1",
            "3": "B2",
            "4": "B3",
            "5": "B3",
            "6": "B3",
            "7": "B3",
            "8": "B4",
            "8A": "B4",
            "11": "B5",
            "12": "B7",
        },
        "S2B": {
            "1": "B1",
            "2": "B1",
            "3": "B2",
            "4": "B3",
            "5": "B3",
            "6": "B3",
            "7": "B3",
            "8": "B4",
            "8A": "B4",
            "11": "B5",
            "12": "B7",
        },
        "S2C": {
            "1": "B1",
            "2": "B1",
            "3": "B2",
            "4": "B3",
            "5": "B3",
            "6": "B3",
            "7": "B3",
            "8": "B4",
            "8A": "B4",
            "11": "B5",
            "12": "B7",
        },
    },
    "LC8": {
        "LS5": {
            "1": "B2",
            "2": "B3",
            "3": "B4",
            "4": "B5",
            "5": "B6",
            "6": "B10",
            "7": "B7",
        },
        "LS7": {
            "1": "B2",
            "2": "B3",
            "3": "B4",
            "4": "B5",
            "5": "B6",
            "61": "B10",
            "62": "B10",
            "7": "B7",
        },
        "LS8": {
            "1": "B1",
            "2": "B2",
            "3": "B3",
            "4": "B4",
            "5": "B5",
            "6": "B6",
            "7": "B7",
            "10": "B10",
            "11": "B11",
        },
        "LS9": {
            "1": "B1",
            "2": "B2",
            "3": "B3",
            "4": "B4",
            "5": "B5",
            "6": "B6",
            "7": "B7",
            "10": "B10",
            "11": "B11",
        },
        "S2A": {
            "1": "B1",
            "2": "B2",
            "3": "B3",
            "4": "B4",
            "5": "B4",
            "6": "B4",
            "7": "B4",
            "8": "B5",
            "8A": "B5",
            "11": "B6",
            "12": "B7",
        },
        "S2B": {
            "1": "B1",
            "2": "B2",
            "3": "B3",
            "4": "B4",
            "5": "B4",
            "6": "B4",
            "7": "B4",
            "8": "B5",
            "8A": "B5",
            "11": "B6",
            "12": "B7",
        },
        "S2C": {
            "1": "B1",
            "2": "B2",
            "3": "B3",
            "4": "B4",
            "5": "B4",
            "6": "B4",
            "7": "B4",
            "8": "B5",
            "8A": "B5",
            "11": "B6",
            "12": "B7",
        },
    },
    "LO8": {
        "LS5": {
            "1": "B2",
            "2": "B3",
            "3": "B4",
            "4": "B5",
            "5": "B6",
            "6": "B10",
            "7": "B7",
        },
        "LS7": {
            "1": "B2",
            "2": "B3",
            "3": "B4",
            "4": "B5",
            "5": "B6",
            "61": "B10",
            "62": "B10",
            "7": "B7",
        },
        "LS8": {
            "1": "B1",
            "2": "B2",
            "3": "B3",
            "4": "B4",
            "5": "B5",
            "6": "B6",
            "7": "B7",
            "10": "B10",
            "11": "B11",
        },
        "LS9": {
            "1": "B1",
            "2": "B2",
            "3": "B3",
            "4": "B4",
            "5": "B5",
            "6": "B6",
            "7": "B7",
            "10": "B10",
            "11": "B11",
        },
        "S2A": {
            "1": "B1",
            "2": "B2",
            "3": "B3",
            "4": "B4",
            "5": "B4",
            "6": "B4",
            "7": "B4",
            "8": "B5",
            "8A": "B5",
            "11": "B6",
            "12": "B7",
        },
        "S2B": {
            "1": "B1",
            "2": "B2",
            "3": "B3",
            "4": "B4",
            "5": "B4",
            "6": "B4",
            "7": "B4",
            "8": "B5",
            "8A": "B5",
            "11": "B6",
            "12": "B7",
        },
        "S2C": {
            "1": "B1",
            "2": "B2",
            "3": "B3",
            "4": "B4",
            "5": "B4",
            "6": "B4",
            "7": "B4",
            "8": "B5",
            "8A": "B5",
            "11": "B6",
            "12": "B7",
        },
    },
}


OLD_BAND_MAP = {
    "LS5": {
        "1": "10",
        "2": "20",
        "3": "30",
        "4": "40",
        "5": "50",
        "6": "61",
        "7": "70",
    },
    "LS7": {
        "1": "10",
        "2": "20",
        "3": "30",
        "4": "40",
        "5": "50",
        "61": "61",
        "62": "62",
        "7": "70",
    },
    "LS8": {
        "1": "10",
        "2": "10",
        "3": "20",
        "4": "30",
        "5": "40",
        "6": "50",
        "7": "70",
        "10": "61",
        "11": "61",
    },
    "LS9": {
        "1": "10",
        "2": "10",
        "3": "20",
        "4": "30",
        "5": "40",
        "6": "50",
        "7": "70",
        "10": "61",
        "11": "61",
    },
    "S2A": {
        "1": "10",
        "2": "10",
        "3": "20",
        "4": "30",
        "5": "30",
        "6": "30",
        "7": "30",
        "8": "40",
        "8A": "40",
        "11": "50",
        "12": "70",
    },
    "S2B": {
        "1": "10",
        "2": "10",
        "3": "20",
        "4": "30",
        "5": "30",
        "6": "30",
        "7": "30",
        "8": "40",
        "8A": "40",
        "11": "50",
        "12": "70",
    },
    "S2C": {
        "1": "10",
        "2": "10",
        "3": "20",
        "4": "30",
        "5": "30",
        "6": "30",
        "7": "30",
        "8": "40",
        "8A": "40",
        "11": "50",
        "12": "70",
    },
}


def _clean_name(s: str) -> str:
    """>>> _clean_name("Residual x ")
    'residual_x'.
    """
    return str(s).strip().lower().replace(" ", "_")


def reproject(
    source_fname: str,
    reference_fname: str,
    out_fname: str,
    cwd: str,
    resampling: Resampling = Resampling.bilinear,
) -> None:
    """Reproject an image.

    :param source_fname:
        A `string` representing the filepath name of the source image.

    :param reference_fname:
        A `string` representing the filepath name of the reference image.

    :param out_fname:
        A `string` representing the filepath name of the output image.

    :param cwd:
        Current working directory to execute `gdalwarp` in.

    :param resampling:
        The resampling method to use during image re-projection.
        Defaults to `bilinear`.
        See rasterio.warp.Resampling for options.

    :notes:
        Just a wrapper for command line GDAL, as the initial testing
        of in-memory vs GDAL command line, failed.
        TODO re-evaluate using a more recent version of rasterio
    """
    with rasterio.open(reference_fname) as ds:
        crs = ds.crs.wkt
        res = ds.res

    # extract the resampling string identifier
    resampling_method = {0: "near", 1: "bilinear", 2: "cubic"}
    resampling = resampling_method[resampling]

    cmd = [
        "gdalwarp",
        "-r",
        resampling,
        "-t_srs",
        f"{crs}",
        "-srcnodata",
        "0",
        "-dstnodata",
        "0",
        "-tr",
        f"{res[0]}",
        f"{res[1]}",
        "-tap",
        "-tap",
        source_fname,
        out_fname,
    ]

    _LOG.info("calling gdalwarp:\n%s", cmd)
    run_command(cmd, cwd)


def _populate_nan_residuals() -> typing.Dict:
    """Returns default values for GQA results."""
    empty_points = {"x": np.nan, "y": np.nan, "xy": np.nan}

    residuals = {
        "mean": empty_points.copy(),
        "stddev": empty_points.copy(),
        "iterative_mean": empty_points.copy(),
        "iterative_stddev": empty_points.copy(),
        "abs_iterative_mean": empty_points.copy(),
        "abs": empty_points.copy(),
        "cep90": np.nan,
    }

    return residuals


def _gls_version(ref_fname: str) -> str:
    """Placeholder implementation for deducing GLS collection version
    TODO: update with a methodical approach.
    """
    if "GLS2000_GCP_SCENE" in ref_fname:
        gls_version = "GLS_v1"
    else:
        gls_version = "GQA_v3"

    return gls_version


def _write_gqa_yaml(out_fname: str, data: typing.Dict) -> None:
    """Writes out the gqa datasets."""
    _LOG.debug("Writing result yaml: %s", out_fname)
    with open(out_fname, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, indent=4)


def _rounded(d: typing.SupportsFloat) -> float:
    """Rounds argument to 2 decimal places."""
    return round(float(d), 2)
