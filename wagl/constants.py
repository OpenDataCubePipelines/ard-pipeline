"""Constants
---------.
"""
# pylint: disable=attribute-defined-outside-init

import re
from enum import Enum
from os.path import join as pjoin

POINT_FMT = "POINT-{p}"
ALBEDO_FMT = "ALBEDO-{a}"
POINT_ALBEDO_FMT = "".join([POINT_FMT, "-", ALBEDO_FMT])


class Workflow(Enum):
    """Represents the different workflow that wagl can run.

    *standard* Indicates both NBAR and SBT workflows will run
    *nbar* Indicates NBAR only
    *sbt* Indicates SBT only
    """

    STANDARD = 1
    NBAR = 2
    SBT = 3

    @property
    def atmos_coefficients(self):
        """Returns the atmospheric coefficients names used for interpolation
        for a given Workflow.<option>.
        """
        atmos_var = list(AtmosphericCoefficients)
        fmap = {
            Workflow.STANDARD: atmos_var,
            Workflow.NBAR: atmos_var[0:8],
            Workflow.SBT: atmos_var[8:],
        }
        return fmap.get(self)

    @property
    def albedos(self):
        """Returns the albedo names used for specific Atmospheric
        evaluations for a given Workflow.<option>.
        """
        albs = list(Albedos)
        amap = {
            Workflow.STANDARD: albs,
            Workflow.NBAR: albs[0:-1],
            Workflow.SBT: [albs[-1]],
        }
        return amap.get(self)

    @property
    def ard_products(self):
        """Returns the ARD products available for a given
        Workflow.<option>.
        """
        products = list(ArdProducts)
        amap = {
            Workflow.STANDARD: products,
            Workflow.NBAR: products[0:-1],
            Workflow.SBT: [products[-1]],
        }
        return amap.get(self)


class BandType(Enum):
    """Represents the Band Type a given acquisition falls under."""

    REFLECTIVE = 0
    THERMAL = 1
    PANCHROMATIC = 2
    ATMOSPHERE = 3
    QUALITY = 4


class DatasetName(Enum):
    """Defines the dataset names or format descriptors, that are used
    for creating and accessing throughout the code base.
    """

    # wagl.ancillary
    COORDINATOR = "COORDINATOR"
    DEWPOINT_TEMPERATURE = "DEWPOINT-TEMPERATURE"
    TEMPERATURE_2M = "TEMPERATURE-2METRE"
    SURFACE_PRESSURE = "SURFACE-PRESSURE"
    SURFACE_GEOPOTENTIAL = "SURFACE-GEOPOTENTIAL-HEIGHT"
    SURFACE_RELATIVE_HUMIDITY = "SURFACE-RELATIVE-HUMIDITY"
    GEOPOTENTIAL = "GEO-POTENTIAL"
    TEMPERATURE = "TEMPERATURE"
    RELATIVE_HUMIDITY = "RELATIVE-HUMIDITY"
    ATMOSPHERIC_PROFILE = "ATMOSPHERIC-PROFILE"
    AEROSOL = "AEROSOL"
    WATER_VAPOUR = "WATER-VAPOUR"
    OZONE = "OZONE"
    ELEVATION = "ELEVATION"
    BRDF_FMT = "BRDF-{parameter}-{band_name}"
    ECMWF_PATH_FMT = pjoin("{product}", "{year}", "tif", "{product}_*.tif")

    # wagl.longitude_latitude_arrays
    LON = "LONGITUDE"
    LAT = "LATITUDE"

    # wagl.satellite_solar_angles
    SATELLITE_VIEW = "SATELLITE-VIEW"
    SATELLITE_AZIMUTH = "SATELLITE-AZIMUTH"
    SOLAR_ZENITH = "SOLAR-ZENITH"
    SOLAR_ZENITH_CHANNEL = "SOLAR-ZENITH-CHANNEL"
    SOLAR_AZIMUTH = "SOLAR-AZIMUTH"
    RELATIVE_AZIMUTH = "RELATIVE-AZIMUTH"
    TIME = "TIME-DELTA"
    CENTRELINE = "CENTRELINE"
    BOXLINE = "BOXLINE"
    SPHEROID = "SPHEROID"
    ORBITAL_ELEMENTS = "ORBITAL-ELEMENTS"
    SATELLITE_MODEL = "SATELLITE-MODEL"
    SATELLITE_TRACK = "SATELLITE-TRACK"
    GENERIC = "GENERIC"

    # wagl.incident_exiting_angles
    INCIDENT = "INCIDENT-ANGLE"
    AZIMUTHAL_INCIDENT = "AZIMUTHAL-INCIDENT"
    EXITING = "EXITING-ANGLE"
    AZIMUTHAL_EXITING = "AZIMUTHAL-EXITING"
    RELATIVE_SLOPE = "RELATIVE-SLOPE"

    # wagl.reflectance
    REFLECTANCE_FMT = "REFLECTANCE/{product}/{band_name}"

    # wagl.temperature
    TEMPERATURE_FMT = "THERMAL/{product}/{band_name}"

    # wagl.terrain_shadow_masks
    SELF_SHADOW = "SELF-SHADOW"
    CAST_SHADOW_FMT = "CAST-SHADOW-{source}"
    COMBINED_SHADOW = "COMBINED-TERRAIN-SHADOW"

    # wagl.slope_aspect
    SLOPE = "SLOPE"
    ASPECT = "ASPECT"

    # wagl.dsm
    DSM = "DSM"
    DSM_SMOOTHED = "DSM-SMOOTHED"

    # wagl.interpolation
    INTERPOLATION_FMT = "{coefficient}/{band_name}"

    # wagl.modtran
    MODTRAN_INPUT = "MODTRAN-INPUT-DATA"
    FLUX = "FLUX"
    ALTITUDES = "ALTITUDES"
    SOLAR_IRRADIANCE = "SOLAR-IRRADIANCE"
    UPWARD_RADIATION_CHANNEL = "UPWARD-RADIATION-CHANNEL"
    DOWNWARD_RADIATION_CHANNEL = "DOWNWARD-RADIATION-CHANNEL"
    CHANNEL = "CHANNEL"
    NBAR_COEFFICIENTS = "NBAR-COEFFICIENTS"
    SBT_COEFFICIENTS = "SBT-COEFFICIENTS"

    # metadata
    METADATA = "METADATA"
    CURRENT_METADATA = "CURRENT"
    NBAR_YAML = "METADATA/NBAR-METADATA"
    SBT_YAML = "METADATA/SBT-METADATA"


class GroupName(Enum):
    """Defines the group names or format descriptors, that are used
    for creating and accessing throughout the code base.
    """

    LON_LAT_GROUP = "LONGITUDE-LATITUDE"
    SAT_SOL_GROUP = "SATELLITE-SOLAR"
    ANCILLARY_GROUP = "ANCILLARY"
    ANCILLARY_AVG_GROUP = "AVERAGED-ANCILLARY"
    ATMOSPHERIC_INPUTS_GRP = "ATMOSPHERIC-INPUTS"
    ATMOSPHERIC_RESULTS_GRP = "ATMOSPHERIC-RESULTS"
    COEFFICIENTS_GROUP = "ATMOSPHERIC-COEFFICIENTS"
    INTERP_GROUP = "INTERPOLATED-ATMOSPHERIC-COEFFICIENTS"
    ELEVATION_GROUP = "ELEVATION"
    SLP_ASP_GROUP = "SLOPE-ASPECT"
    INCIDENT_GROUP = "INCIDENT-ANGLES"
    EXITING_GROUP = "EXITING-ANGLES"
    REL_SLP_GROUP = "RELATIVE-SLOPE"
    SHADOW_GROUP = "SHADOW-MASKS"
    STANDARD_GROUP = "STANDARDISED-PRODUCTS"


class Method(Enum):
    """Defines the Interpolation method used for interpolating the
    atmospheric coefficients.
    """

    BILINEAR = 0
    FBILINEAR = 1
    SHEAR = 2
    SHEARB = 3


class BrdfModelParameters(Enum):
    """Defines the BRDF Parameters used in BRDF correction."""

    ISO = "ISO"
    VOL = "VOL"
    GEO = "GEO"


class BrdfDirectionalParameters(Enum):
    """Defines the BRDF Parameters used in BRDF correction."""

    ALPHA_1 = "ALPHA-1"
    ALPHA_2 = "ALPHA-2"


class ArdProducts(Enum):
    """Defines the output ARD products that wagl produces."""

    NBAR = "NBAR"
    NBART = "NBART"
    LAMBERTIAN = "LAMBERTIAN"
    SBT = "SBT"


class Albedos(Enum):
    """Defines the albedo labels that wagl uses."""

    ALBEDO_0 = "0"
    ALBEDO_TH = "TH"


class AtmosphericCoefficients(Enum):  # param, coeff, vari... what to use
    """Defines the atmospheric coefficient names that wagl uses."""

    FS = "FS"
    FV = "FV"
    A = "A"
    B = "B"
    S = "S"
    DIR = "DIR"
    DIF = "DIF"
    TS = "TS"
    PATH_UP = "PATH-UP"
    PATH_DOWN = "PATH-DOWN"
    TRANSMITTANCE_UP = "TRANSMITTANCE-UP"
    ESUN = "ESUN"


class TrackIntersection(Enum):
    """Defines the type of track intersection an acquisition
    will have.
    """

    FULL = 0
    PARTIAL = 1
    EMPTY = 2


class WaterVapourTier(Enum):
    """Defines the tier levels for the water vapour data.
    The higher the value, the higher the precedence.
    """

    FALLBACK_DEFAULT = 0  # default value if everything fails
    FALLBACK_DATASET = 1  # averages for each hourly period in the archive
    DEFINITIVE = 2  # value taken from date of acquisition
    USER = 3  # user specified


class BrdfTier(Enum):
    """Defines the tier levels for the BRDF data.
    The higher the value, the higher the precedence.
    """

    FALLBACK_DEFAULT = 0  # default value if everything fails
    FALLBACK_DATASET = 1  # averages for each day of year over the entire archive
    DEFINITIVE = 2  # value taken from date of acquisition
    USER = 3  # user specified


class AerosolTier(Enum):
    """Defines the tier levels for the aerosol data.
    The higher the value, the higher the precedence.
    """

    FALLBACK_DEFAULT = 0  # default value is everything fails
    AATSR_CMP_MONTH = 1  # monthly composites for all years i.e. jun 2002, jun 2003
    AATSR_CMP_YEAR_MONTH = 2  # composite for all data within a given month and year
    AATSR_PIX = 3  # value taken from date of acquisition
    USER = 4  # user specified


class OzoneTier(Enum):
    """Defines the tier levels for the ozone data.
    The higher the value, the higher the precedence.
    """

    DEFINITIVE = 0  # value taken from dataset
    USER = 1  # user specified


def combine_satellite_sensor(satellite, sensor):
    """A small utility to deal with GA's and USGS's various naming
    conventions.
    This joins the two strings into an ugly looking string.
    """
    # NOTE: GA Landsat products use both '-' and '_' as a seperator
    # Remove any occurences of - and _ then convert to lowercase
    satellite_name = re.sub("[-_]", "", satellite).lower()
    sensor_name = re.sub("[-_]", "", sensor).lower()
    return "".join((satellite_name, sensor_name))


def sbt_bands(satellite, sensor):
    """Retrieve the thermal bands to be processed through to SBT for
    a given satellite sensor.
    """
    combined = combine_satellite_sensor(satellite, sensor)

    lookup = {
        "landsat5tm": ["6"],
        "landsat7etm+": ["61", "62"],
        "landsat8olitirs": ["10"],
    }  # band 11 is not stable

    return lookup.get(combined, [])
