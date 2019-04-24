from scimath.units.api import UnitScalar, UnitArray, convert, has_units
from scipy.constants import pi

import numpy as np
###################################################################
#                       Total cross section                       #
###################################################################

#### 1a) Elastic Rutherford scattering cross section
@has_units
def ruther_sigma(E, Z):
    """ Calculate the Rutherford elastic cross section
        per atom

        Parameters
        ----------
        E      : array : units = KeV

        Z      : array : units = dim

        Returns
        -------
        s_R    : array : units = cm**2
    """
    alpha =  3.4e-3*(Z**(0.67))/E
    s_R = 5.21e-21 * (Z**2/(E**2)) * (4.*pi)/(alpha*(1. + alpha)) * ((E + 511.)/(E + 1024.))**2
    return s_R

#### 2a) Inelastic Moller cross section for free electrons
@has_units
def moller_sigma(E, Emin, nfree, c_pi_efour):
    """ Calculate the Moller inelastic cross section
        per atom

        Parameters
        ----------
        E      : array : units = eV

        Emin   : array : units = eV

        nfree  : array : units = dim

        c_pi_efour: scalar: units = cm**2 * eV**2

        Returns
        -------
        s_M    : array : units = cm**2
    """

    eps = Emin*1./E
    s_M = nfree*c_pi_efour*(1./(E*E)) * (1./eps - 1./(1.-eps) + np.log(eps/(1.-eps)))
    return s_M


#### 2b) Inelastic Gryzinski cross section for core shell electrons
@has_units
def gryz_sigma(E, Enl, nsi, c_pi_efour):
    """ Calculate the Moller inelastic cross section
        per atom

        Parameters
        ----------
        E      : array : units = eV

        Enl    : array : units = eV

        nsi    : array : units = dim
                 number of electrons in i-shell
        c_pi_efour: scalar: units = cm**2 * eV**2

        Returns
        -------
        s_G    : array : units = cm**2
    """

    U = E*1./Enl
    s_G = nsi * c_pi_efour * ((U - 1.)/(U + 1.))**1.5 * (1. + (2.*(1.-(1./(2.*U)))/3. *
              np.log(2.7 + ((U-1.)**0.5))) ) /(E*Enl)
    return s_G


#### 2c) Inelastic Quinn cross section for plasmons
@has_units
def quinn_sigma(E, Epl, Ef, n, bohr_r):
    """ Calculate the Quinn inelastic cross section
        per atom

        Parameters
        ----------
        E      : array : units = eV
                incident energy

        Epl    : array : units = eV
                plasmon energy

        Ef     : array : units = eV
                Fermi energy

        n      : array : units = m**-3
               number density

        bohr_r :scalar : units = m

        Returns
        -------
        s_Q    : array : units = cm**2
    """
    E_Ef = E*1./Ef
    Epl_Ef = Epl*1./Ef

    s_Q_total = Epl*np.log( ((1. + Epl_Ef)**(0.5) - 1.)/ ( E_Ef**0.5 -
             (E_Ef - Epl_Ef)**0.5 ) )/(2. *  bohr_r * E)
    s_Q = s_Q_total/n * m**2/ cm**2

    return s_Q
