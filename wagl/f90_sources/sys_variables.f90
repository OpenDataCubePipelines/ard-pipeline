module sys_variables
    implicit none

    public pi, d2r, r2d, arcsecond
    double precision, parameter :: pi = 4.0d0*atan(1.0d0)
    double precision, parameter :: d2r = pi/180.0d0
    double precision, parameter :: r2d = 180.0d0/pi
    double precision, parameter :: arcsecond = 1.0d0 / 3600

end module sys_variables
