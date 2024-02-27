! subroutine angle
SUBROUTINE satellite_angle(nrow,ncol,nlines,row_offset,col_offset,alat,alon,spheroid,orb_elements, &
             ntpoints,smodel,track, &
             view,azi,tim,X_cent,N_cent,istat)

!   program to calculate solar, view and azimuth angle from
!   both UTM and lat/lon projection if we only know one point in the
!   satellite track. the program is written by Fuqin Li, Mar., 2014.
!   the subroutines cal_angle are from David Jupp

!   * Re-written as an F2Py subroutine by JS, Aug 2014

!   Inputs:
!       nrow
!       ncol
!       nlines
!       row_offset
!       col_offset
!       alat
!       alon
!       spheroid
!           1. Spheroid major axis
!           2. Inverse flattening
!           3. Eccentricity squared
!           4. Earth rotational angular velocity rad/sec
!       orb_elements
!           1. Orbital inclination (degrees)
!           2. Semi_major radius (m)
!           3. Angular velocity (rad sec-1)
!       ntpoints (number of time points created in determining the satellite track)
!       smodel
!           1. phi0
!           2. phi0_p
!           3. rho0
!           4. t0
!           5. lam0
!           6. gamm0
!           7. beta0
!           8. rotn0
!           9. hxy0
!           10. N0
!           11. H0
!           12. th_ratio0
!       track
!           1. t
!           2. rho
!           3. phi_p
!           4. lam
!           5. beta
!           6. hxy
!           7. mj
!           8. skew
!       view
!       azi
!       tim
!       X_cent
!       N_cent
!
!   Outputs
!       view
!       azi
!       tim
!       X_cent
!       N_cent
!       istat

    use sys_variables, only : pi, d2r, r2d

    implicit none

    integer, intent(in) :: nrow, ncol, nlines, row_offset, col_offset
    double precision, dimension(nrow, ncol), intent(in) :: alat, alon
    double precision, dimension(4), intent(in) :: spheroid
    double precision, dimension(3), intent(in) :: orb_elements
    integer, intent(in) :: ntpoints
    double precision, dimension(12), intent(in) :: smodel
    double precision, dimension(ntpoints,8), intent(in) :: track
    real, dimension(nrow, ncol), intent(inout) :: view, azi, tim
    real, dimension(nlines), intent(inout) :: X_cent, N_cent
    integer, dimension(nrow, ncol), intent(out) :: istat
    double precision delxx, tol_lam
    double precision xout, yout
    double precision phip_p, lam_p
    double precision timet, theta_p, azimuth
    double precision, dimension(ntpoints) :: tan_beta
    double precision orb_incl, sin_orb_incl, cos_orb_incl, tan_orb_incl
    integer i, j, istat_elem

!f2py depend(nrow, ncol), alat, alon, view, azi, tim
!f2py depend(nlines), X_cent, N_cent
!f2py depend(ntpoints), track

!   Orbital inclination
    orb_incl = orb_elements(1)*d2r
    sin_orb_incl = sin(orb_incl)
    cos_orb_incl = cos(orb_incl)
    tan_orb_incl = tan(orb_incl)

    do i=1,ntpoints
        tan_beta(i) = tan(track(i, 5))
    end do

    do i=1,nrow
    do j=1,ncol
        xout = alon(i, j)
        yout = alat(i, j)

!       calculate pixel size (half)
        if (j .gt. 1) then
            delxx = (alon(i, j)-alon(i, j-1))/2
        else
            delxx = (alon(i, j+1)-alon(i, j))/2
        endif

        tol_lam = delxx*d2r*1.2

!       go through the base sequence used in the test examples
        lam_p = xout*d2r

        call geod2geo(yout, orb_elements, spheroid, phip_p, istat_elem)

        istat(i, j) = istat_elem

        call cal_angles(lam_p, phip_p, tol_lam, orb_elements, &
               spheroid, smodel, track, tan_beta, &
               sin_orb_incl, cos_orb_incl, tan_orb_incl, &
               ntpoints, timet, theta_p, &
               azimuth, istat_elem)

        if (istat(i, j) .eq. 0) istat(i, j) = istat_elem

        tim(i, j) = timet
        view(i, j) = theta_p*r2d
        azi(i, j) = azimuth*r2d
        if ((abs(timet) .gt. 1.0e-5) .and. (abs(view(i, j)) .lt. 1.0e-7) &
          .and. (abs(azi(i, j)) .lt. 1.0e-7)) then
            X_cent(row_offset + i) = X_cent(row_offset + i)+real(j + col_offset)
            N_cent(row_offset + i) = N_cent(row_offset + i)+1.0
        endif
    enddo
    enddo

    return

END SUBROUTINE satellite_angle
