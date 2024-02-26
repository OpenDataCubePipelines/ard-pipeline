! subroutine solar
SUBROUTINE solar(alat,along,tt,sun_declin,eq_time,solar_zen,sazi)
!   program used to calculate solar angle using NOAA method
!   it is more accurate and using julian day
!   the program is written by Fuqin Li in 2010

!   * Re-written as an indepentent subroutine by JS, Aug 2014

!   Inputs:
!       alat
!       along
!       tt
!       sun_declin
!       eq_time

!   Outputs:
!       solar_zen
!       sazi

    use sys_variables, only : pi, d2r, r2d

    implicit none

    double precision, intent(in) :: alat, along
    double precision, intent(in) :: tt
    double precision, intent(in) :: sun_declin
    double precision, intent(in) :: eq_time
    real, intent(out) :: solar_zen, sazi

    double precision suntime_true, hangle
    double precision solar_zen1, solar_ele1, atmo
    double precision solar_ele, sss, ss_time
    double precision sin_alat, cos_alat
    double precision sin_sun_declin, cos_sun_declin
    double precision tan_solar_ele1

    ss_time = tt*60+eq_time+4*along*r2d
    suntime_true = dmod(ss_time,1440d0)

    if (suntime_true/4.0 .ge. 0) then
        hangle = suntime_true/4.0-180
    else
        hangle = suntime_true/4.0+180
    endif

    sin_alat = sin(alat)
    cos_alat = cos(alat)

    sin_sun_declin = sin(sun_declin)
    cos_sun_declin = cos(sun_declin)

    solar_zen1 = r2d*(acos(sin_alat*sin_sun_declin+ &
      cos_alat*cos_sun_declin*cos(d2r*hangle)))

    solar_ele1 = 90.0-solar_zen1

    if (solar_ele1 .gt. 85) atmo = 0

    if (solar_ele1 .gt. 5 .and. solar_ele1 .le.85) then
      tan_solar_ele1 = tan(solar_ele1*d2r)
      atmo = 58.1/tan_solar_ele1-0.07/(tan_solar_ele1**3)+0.000086d0/ &
      (tan_solar_ele1**5)
    end if

    if (solar_ele1 .gt. -0.575 .and. solar_ele1 .le.5) atmo = &
      1735.0+solar_ele1*(-518.2+solar_ele1*(103.4+solar_ele1*(-12.79+ &
      solar_ele1*0.711)))

    if (solar_ele1 .le. -0.575 ) atmo = -20.772/tan(solar_ele1*d2r)

    atmo = atmo/3600.0
    solar_ele = solar_ele1+atmo
    solar_zen = 90-solar_ele

    if (hangle .gt. 0) then
        sss = r2d*(acos(((sin_alat*cos(d2r*solar_zen1))-sin_sun_declin) &
            /(cos_alat*sin(d2r*solar_zen1))))+180.0
    else
        sss = 540.0-r2d*(acos(((sin_alat*cos(d2r*solar_zen1))- &
          sin_sun_declin)/(cos_alat*sin(d2r*solar_zen1))))
    endif

    sazi = dmod(sss,360d0)

    return

END SUBROUTINE solar
