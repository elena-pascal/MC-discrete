from scimath.units.api import UnitScalar, UnitArray, convert, has_units
import numpy as np

###################################################################
#                       Excitation function                       #
###################################################################

# Moller free electron discrete cross section
@has_units
def moller_dCS(E, W, nfree, c_pi_efour):
    """ Calculate the Moller inelastic cross section

        Parameters
        ----------
        E      : array : units = eV
                       incident energy

        W      : array : units = eV
                       energy loss

        nfree  : array : units = dim

        c_pi_efour: scalar: units = cm**2 * eV**2

        Returns
        -------
        dCS_M  : array : units = cm**2/eV
    """
    eps = W*1./E
    if (((1.-eps)**2) > 0.): # 1-eps can be very small
        dCS_M =  nfree*c_pi_efour* E**2 *( 1./(eps**2) +
                 ( 1./((1.-eps)**2) ) - ( 1./(eps*(1.-eps)) ) )
    else:
        print '1-eps very small in Moller discrete CS'
        dCS_M = 0.
    return dCS_M


# 2b) Gryzinski differential cross section for core shell electrons
@has_units
def gryz_dCS(E, W, nsi, c_pi_efour, Ebi):
    """ Calculate the Moller inelastic cross section

        Parameters
        ----------
        E      : array : units = eV
                       incident energy

        W      : array : units = eV
                       energy loss

        Ebi    : array : units = eV
                       binding energy of shell i

        nsi    : array : units = dim
                       number of electrons in shell i

        c_pi_efour: scalar: units = cm**2 * eV**2

        Returns
        -------
        dCS_G    : array : units = cm**2
    """

    eps = W*1./E
    epsB = Ebi*1./E

    if ((1. - eps)/epsB **(0.5) > 0): # 1-eps can be very small
        dCS_G = nsi * c_pi_efour * (1. + epsB)**(-1.5) * (1. - eps)**(epsB/(epsB+eps)) * ((1. - epsB) +
                               4. * epsB * np.log(2.7 + ((1. - eps)/epsB)**(0.5) )/3. )   /(eps**2 * E**2)
    else:
        dCS_G = 0.
    return dCS_G
