SUBROUTINE cal_pole(theta, phi, theta_p, phi_p, thp, php)
! does some funky stuff to calculate various angles
    real, intent(in) :: theta, phi, theta_p, phi_p
    real, intent(inout) :: thp, php
    real offset
    real pdiff, costhp, sinphp, cosphp, tnum, tden
    real cos_theta, sin_theta, cos_theta_p, sin_theta_p
    real cos_pdiff, sin_pdiff
    real sin_thp
    real eps

    eps=1.0e-6
    pi=4.0*atan(1.0)
    d2r=pi/180.0

    if (abs(theta_p).le.eps) then
        thp=theta
        php=phi
        return
    endif

    cos_theta = cos(d2r*theta)
    sin_theta = sin(d2r*theta)
    cos_theta_p = cos(d2r*theta_p)
    sin_theta_p = sin(d2r*theta_p)

    offset = atan(tan(pi-d2r*phi_p)*cos_theta_p)
    pdiff = d2r*(phi-phi_p)
    cos_pdiff = cos(pdiff)
    sin_pdiff = sin(pdiff)

    costhp = cos_theta*cos_theta_p+sin_theta*sin_theta_p*cos_pdiff

    if (costhp.ge.1.0-eps) then
        thp=0.0
        php=0.0-offset/d2r
        return
    else
        thp=acos(costhp)/d2r
        sin_thp = sin(d2r*thp)
        sinphp=sin_theta*sin_pdiff/sin_thp
        cosphp=(cos_theta*sin_theta_p-sin_theta*cos_theta_p*cos_pdiff)/sin_thp
        if (abs(sinphp).le.eps) then
            if (cosphp.gt.eps) then
                php=0.0-offset/d2r
            else
                php=180.0-offset/d2r
            endif
            return
        else if (abs(cosphp).le.eps) then
            if (sinphp.gt.eps) then
                php=90.0-offset/d2r
            else
                php=-90.0-offset/d2r
            endif
            return
        endif
    endif

    tnum=sin_theta*sin_pdiff
    tden=(cos_theta*sin_theta_p-sin_theta*cos_theta_p*cos_pdiff)
    php=(atan2(tnum,tden)-offset)/d2r
    return
END SUBROUTINE cal_pole
