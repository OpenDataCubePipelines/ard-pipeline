"""
TODO: Module for atmospheric calculations.
"""

import numpy as np


def kelvin_2_celcius(kelvin):
    """A small utility function for converting degrees Kelvin to
    degrees Celcius.
    """
    return kelvin - 273.15


def relative_humdity(surface_temp, dewpoint_temp, kelvin=True, clip_negative=True):
    """
    Calculates relative humidity given a surface temperature & dewpoint temperature.

    MODTRAN doesn't handle negative RH values, thus the default behaviour is to
    clip negative values to zero to sanitise calculation.
    """
    if kelvin:
        surf_t = kelvin_2_celcius(surface_temp)
        dew_t = kelvin_2_celcius(dewpoint_temp)
    else:
        surf_t = surface_temp
        dew_t = dewpoint_temp

    rh = 100 * ((112.0 - 0.1 * surf_t + dew_t) / (112.0 + 0.9 * surf_t)) ** 8

    if clip_negative:
        rh = np.where(rh < 0.0, 0.0, rh)

    return rh
