#!/usr/bin/env python

"""Unit test for vincenty.py."""

import math
import unittest

from wagl.vincenty import GreatCircle, vinc_dist, vinc_pt


class VincentyTestCase(unittest.TestCase):
    def runTest(self):
        # WGS84
        a = 6378137.0
        b = 6356752.3142
        f = (a - b) / a

        print(f"\n Ellipsoidal major axis =  {a:12.3f} metres\n")
        print(f"\n Inverse flattening     =  {1.0 / f:15.9f}\n")

        print("\n Test Flinders Peak to Buninyon")
        print("\n ****************************** \n")
        phi1 = -((3.7203 / 60.0 + 57) / 60.0 + 37)
        lembda1 = (29.5244 / 60.0 + 25) / 60.0 + 144
        print(f"\n Flinders Peak = {phi1:12.6f}, {lembda1:13.6f} \n")
        deg = int(phi1)
        minn = int(abs((phi1 - deg) * 60.0))
        sec = abs(phi1 * 3600 - deg * 3600) - minn * 60
        print(f" Flinders Peak =   {deg:3d}\xf8{minn:3d}' {sec:6.3f}\",  ", end=" ")
        deg = int(lembda1)
        minn = int(abs((lembda1 - deg) * 60.0))
        sec = abs(lembda1 * 3600 - deg * 3600) - minn * 60
        print(f" {deg:3d}\xf8{minn:3d}' {sec:6.3f}\" \n")

        phi2 = -((10.1561 / 60.0 + 39) / 60.0 + 37)
        lembda2 = (35.3839 / 60.0 + 55) / 60.0 + 143
        print(f"\n Buninyon      = {phi2:12.6f}, {lembda2:13.6f} \n")

        deg = int(phi2)
        minn = int(abs((phi2 - deg) * 60.0))
        sec = abs(phi2 * 3600 - deg * 3600) - minn * 60
        print(f" Buninyon      =   {deg:3d}\xf8{minn:3d}' {sec:6.3f}\",  ", end=" ")
        deg = int(lembda2)
        minn = int(abs((lembda2 - deg) * 60.0))
        sec = abs(lembda2 * 3600 - deg * 3600) - minn * 60
        print(f" {deg:3d}\xf8{minn:3d}' {sec:6.3f}\" \n")

        dist, alpha12, alpha21 = vinc_dist(
            f,
            a,
            math.radians(phi1),
            math.radians(lembda1),
            math.radians(phi2),
            math.radians(lembda2),
        )

        alpha12 = math.degrees(alpha12)
        alpha21 = math.degrees(alpha21)

        print(
            f"\n Ellipsoidal Distance = {dist:15.3f} metres\n"
            f"            should be         54972.271 m\n"
        )
        print(f"\n Forward and back azimuths = {alpha12:15.6f}, {alpha21:15.6f} \n")
        deg = int(alpha12)
        minn = int(abs((alpha12 - deg) * 60.0))
        sec = abs(alpha12 * 3600 - deg * 3600) - minn * 60
        print(" Forward azimuth = %3i\xf8%3i' %6.3f\"\n" % (deg, minn, sec))
        deg = int(alpha21)
        minn = int(abs((alpha21 - deg) * 60.0))
        sec = abs(alpha21 * 3600 - deg * 3600) - minn * 60
        print(" Reverse azimuth = %3i\xf8%3i' %6.3f\"\n" % (deg, minn, sec))

        # Test the direct function */
        phi1 = -((3.7203 / 60.0 + 57) / 60.0 + 37)
        lembda1 = (29.5244 / 60.0 + 25) / 60.0 + 144
        dist = 54972.271
        alpha12 = (5.37 / 60.0 + 52) / 60.0 + 306

        phi2, lembda2, alpha21 = vinc_pt(
            f, a, math.radians(phi1), math.radians(lembda1), math.radians(alpha12), dist
        )

        phi2 = math.degrees(phi2)
        lembda2 = math.degrees(lembda2)
        alpha21 = math.degrees(alpha21)

        print(f"\n Projected point ={phi2:11.6f}, {lembda2:13.6f} \n")
        deg = int(phi2)
        minn = int(abs((phi2 - deg) * 60.0))
        sec = abs(phi2 * 3600 - deg * 3600) - minn * 60
        print(f" Projected Point = {deg:3d}\xf8{minn:3d}' {sec:6.3f}\", ", end=" ")
        deg = int(lembda2)
        minn = int(abs((lembda2 - deg) * 60.0))
        sec = abs(lembda2 * 3600 - deg * 3600) - minn * 60
        print("  %3i\xf8%3i' %6.3f\"\n" % (deg, minn, sec))
        print(" Should be Buninyon \n")
        print(f"\n Reverse azimuth = {alpha21:10.6f} \n")
        deg = int(alpha21)
        minn = int(abs((alpha21 - deg) * 60.0))
        sec = abs(alpha21 * 3600 - deg * 3600) - minn * 60
        print(f" Reverse azimuth = {deg:3d}\xf8{minn:3d}' {sec:6.3f}\"\n\n")

        # lat/lon of New York
        lat1 = 40.78
        lon1 = -73.98
        # lat/lon of London.
        lat2 = 51.53
        lon2 = 0.08
        print("New York to London:")
        gc = GreatCircle((2 * a + b) / 3.0, (2 * a + b) / 3.0, lon1, lat1, lon2, lat2)
        print("geodesic distance using a sphere with WGS84 mean radius = ", gc.distance)
        print("lon/lat for 10 equally spaced points along geodesic:")
        lons, lats = gc.points(10)
        for lon, lat in zip(lons, lats):
            print(lon, lat)
        gc = GreatCircle(a, b, lon1, lat1, lon2, lat2)
        print("geodesic distance using WGS84 ellipsoid = ", gc.distance)
        print("lon/lat for 10 equally spaced points along geodesic:")
        lons, lats = gc.points(10)
        for lon, lat in zip(lons, lats):
            print(lon, lat)


def the_suite():
    """Returns a test suite of all the tests in this module."""
    test_classes = [VincentyTestCase]

    suite_list = map(unittest.defaultTestLoader.loadTestsFromTestCase, test_classes)

    suite = unittest.TestSuite(suite_list)

    return suite


#
# Run unit tests if in __main__
#

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(the_suite())
