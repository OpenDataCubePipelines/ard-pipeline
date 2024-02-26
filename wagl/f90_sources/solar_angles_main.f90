! subroutine angle
SUBROUTINE solar_angle(nrow,ncol,alat,alon, &
             hours,century, &
             asol,soazi)

!   program to calculate solar, view and azimuth angle from
!   both UTM and lat/lon projection if we only know one point in the
!   satellite track. the program is written by Fuqin Li, Mar., 2014.
!   the subroutines cal_angle are from David Jupp

!   * Re-written as an F2Py subroutine by JS, Aug 2014

!   Inputs:
!       nrow
!       ncol
!       alat
!       alon
!       hours
!       century
!       asol
!       soazi
!
!   Outputs
!       asol
!       soazi

    use sys_variables, only : pi, d2r, r2d

    implicit none

    integer, intent(in) :: nrow, ncol
    double precision, dimension(nrow, ncol), intent(in) :: alat, alon
    double precision, intent(in) :: hours, century
    real, dimension(nrow, ncol), intent(inout) :: asol, soazi 
    double precision xout, yout
    double precision phip_p, lam_p
    double precision timet, theta_p, azimuth
    double precision geom_ss, sun_long, sun_anom, eccent, sun_eqc
    double precision sun_truelong, sun_trueanom
    double precision sun_rad, sun_app, ecliptic_mean
    double precision obliq_corr, sun_asc, sun_declin
    double precision vary, eq_time
    integer i, j

!f2py depend(nrow, ncol), alat, alon, asol, soazi
!f2py depend(ntpoints), track

    geom_ss = 280.46646d0+century*(36000.76983d0+century*0.0003032d0)
    sun_long = dmod(geom_ss,360d0)
    sun_anom = 357.52911d0+century*(35999.05029d0-0.0001537d0*century)
    eccent = 0.016708634d0-century*(0.000042037d0+0.0001537d0*century)

    sun_eqc = sin(sun_anom*d2r)*(1.914602d0-century*(0.004817d0+ &
      0.000014d0*century))+sin(2*sun_anom*d2r)*(0.019993d0-0.000101d0* &
      century)+sin(3*sun_anom*d2r)*0.000289d0

    sun_truelong = sun_long+sun_eqc
    sun_trueanom = sun_anom+sun_eqc

    sun_rad = (1.000001018d0*(1-eccent**2))/(1+eccent* &
      cos(sun_trueanom*d2r))

    sun_app = sun_truelong-0.00569d0-0.00478d0*sin(d2r*(125.04- &
      1937.136d0*century))

    ecliptic_mean = 23.0+(26.0+((21.448-century*(46.815+century* &
      (0.00059d0-century*0.001813d0))))/60.0)/60.0

    obliq_corr = ecliptic_mean+0.00256*cos((125.04-1934.136*century)* &
      d2r)

    sun_asc = r2d*(atan2(cos(sun_app*d2r),cos(obliq_corr*d2r)* &
      sin(sun_app*d2r)))

    sun_declin = r2d*(asin(sin(obliq_corr*d2r)*sin(sun_app*d2r)))
    vary = tan(obliq_corr/2.0*d2r)**2

    eq_time = 4*r2d*(vary*sin(2.0*sun_long*d2r)-2*eccent*sin(d2r* &
      sun_anom)+ 4*eccent*vary*sin(d2r*sun_anom)*cos(2*d2r*sun_long) &
      -0.5*vary**2* sin(4*d2r*sun_long)-1.25*eccent**2*sin(2*d2r* &
      sun_anom))

    do i=1,nrow
    do j=1,ncol
        xout = alon(i, j)
        yout = alat(i, j)

!       calculate solar angle
        call solar(yout*d2r, xout*d2r, hours, &
               sun_declin*d2r, eq_time, &
               asol(i, j), soazi(i, j))
    enddo
    enddo

    return

END SUBROUTINE solar_angle
