! subroutine geod2geo
SUBROUTINE geod2geo(yin,orb_elements,spheroid,phip,istat)
!   This routine converts a scalar from geodetic latitude
!   in degrees to geocentric in radians

!   * Re-written as an independent subroutine by JS, Aug 2014

!   Inputs:
!       yin
!       orb_elements
!           1. Orbital inclination (degrees)
!           2. Semi_major radius (m)
!           3. Angular velocity (rad sec-1)
!       spheroid
!           1. Spheroid major axis
!           2. Inverse flattening
!           3. Eccentricity squared
!           4. Earth rotational angular velocity rad/sec

!   Outputs:
!       phip
!       istat

    use sys_variables, only : pi, d2r, r2d

    implicit none

    double precision yin, phip
    double precision orb_elements(3), spheroid(4)
    double precision orad
    double precision asph, e2
    double precision rn, temp, sin_yin, cos_yin
    integer istat

!   Satellite orbital paramaters
!   semi_major radius (m)
    orad = orb_elements(2)

!   Spheroid paramaters
!   Spheroid major axis
!   Eccentricity squared
    asph = spheroid(1)
    e2 = spheroid(3)

!   Initialise the return status
    istat = 0

    temp = dble(yin)*d2r
    sin_yin = sin(temp)
    cos_yin = cos(temp)
    RN = asph/sqrt(1.0d0-e2*sin_yin**2)
    phip = temp-asin(RN*e2*sin_yin*cos_yin/orad)

    return

END SUBROUTINE geod2geo
