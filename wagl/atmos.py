"""
TODO: Module for atmospheric calculations.
"""

import numpy as np

MIN_RELATIVE_HUMIDITY = 0.0
MAX_RELATIVE_HUMIDITY = 100.0


def kelvin_2_celcius(kelvin):
    """A small utility function for converting degrees Kelvin to
    degrees Celcius.
    """
    return kelvin - 273.15


def relative_humidity(surface_temp, dewpoint_temp, kelvin=True, clip_overflow=True):
    """
    Calculates relative humidity given a surface temperature & dewpoint temperature.

    MODTRAN doesn't handle negative RH values, thus the default behaviour is to
    clip over values to zero or 100 to sanitise RH calculations. Negative RH is
    clipped to zero, RH > 100 is clipped to 100.
    """
    if kelvin:
        surf_t = kelvin_2_celcius(surface_temp)
        dew_t = kelvin_2_celcius(dewpoint_temp)
    else:
        surf_t = surface_temp
        dew_t = dewpoint_temp

    rh = 100 * ((112.0 - 0.1 * surf_t + dew_t) / (112.0 + 0.9 * surf_t)) ** 8

    if clip_overflow:
        rh = np.clip(rh, MIN_RELATIVE_HUMIDITY, MAX_RELATIVE_HUMIDITY)

    return rh
