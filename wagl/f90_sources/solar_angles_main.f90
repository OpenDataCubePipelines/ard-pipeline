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
    double precision delxx, tol_lam
    double precision xout, yout
    double precision phip_p, lam_p
    double precision timet, theta_p, azimuth
    integer i, j

!f2py depend(nrow, ncol), alat, alon, asol, soazi
!f2py depend(ntpoints), track

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

!       calculate solar angle
        call solar(yout*d2r, xout*d2r, century, hours, asol(i, j), &
               soazi(i, j))
    enddo
    enddo

    return

END SUBROUTINE solar_angle
