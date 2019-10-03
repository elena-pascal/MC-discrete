#from scimath.units.api import has_units
from numpy import log
import numpy as np
import numpy.ma as ma
import sys
from errors import E_lossTooLarge

from parameters import pi_efour

###################################################################
#                       Excitation functions d_sigma/dW           #
###################################################################
# Probability that an electron of energy E will loose energy W
# in a scattering event

# Since these functions are only used for determining the energy loss
# in the numerical integration, I disabled the units for now
# TODO: do this nicer though

# 2b) Moller free electron discrete cross section
#@has_units
def moller_dCS(E, W, nfree, c_pi_efour=pi_efour):
    """ Calculate the Moller inelastic discrete cross section

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
        dCS  : array : units = cm**2
    """
    eps = W/E

    # check if W arrived here to be smaller than E
    if ( (type(W) is np.float32) or (type(W) is float) ):
        assert (eps<1.0), 'W is larger than E'

    elif ((type(W) is np.ndarray) or (type(W) is ma.core.MaskedArray)):
        assert (np.all(eps<1.0)), 'W is larger than E'
    else:
        print ('W:', type(W))
        sys.exit('W has the wrong type in moller_dCS')


    dCS = nfree*c_pi_efour *( 1./(eps**2) +
                  ( 1./((1.-eps)**2) ) - ( 1./(eps*(1.-eps)) ) )/ E**3
    return dCS


# 2b) Gryzinski differential cross section for core shell electrons
#@has_units
def gryz_dCS(E, W, nsi, Ebi, c_pi_efour=pi_efour):
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
        dCS    : array : units = cm**2
    """

    eps = W/E
    epsB = Ebi/E

    # check if W arrived here to be smaller than E
    if ( (type(W) is np.float32) or (type(W) is float) ):
        assert (eps<1.0), 'W is larger than E'

    elif ((type(W) is np.ndarray) or (type(W) is ma.core.MaskedArray)):
        assert (np.all(eps<1.0)), 'W is larger than E'
    else:
        sys.exit('W has the wrong type in gryz_dCS')

    # check if E is not smaller than Ebi
    if ((type(E) is np.float32) or (type(E) is float)):
        assert (epsB<1.0), 'Ebi larger than E'

    elif ((type(E) is np.ndarray) or (type(E) is ma.core.MaskedArray)):
        assert (np.all(epsB<1.0)), 'Ebi larger than E'
    else:
        sys.exit('E has the wrong type in gryz_dCS')

    dCS = nsi * c_pi_efour * eps * (1. + epsB)**(-1.5) * (1. - eps)**(epsB/(epsB+eps)) * ((1. - epsB) +
                                   4. * epsB * log(2.7 + ((1. - eps)/epsB)**(0.5) )/(3.*eps) )   /( W**3)

    return dCS

# 2b') Gryzinski differential cross section for core shell electrons
# but following Patrik's approach where he sums up all the shells cotributions
#@has_units
def gryz_dCS_P(E, W, nsi, Ebi, c_pi_efour=pi_efour):
    """ Calculate the Moller inelastic cross section

        Parameters
        ----------
        E      : array : units = eV
                       incident energy

        W      : array : units = eV
                       energy loss

        Ebi    : array : units = eV
                       array of binding energy of shell i

        nsi    : array : units = dim
                       array of number of electrons per shell i

        c_pi_efour: scalar: units = cm**2 * eV**2

        Returns
        -------
        dCS    : array : units = cm**2
    """

    dCS = 0.

    for indx, ni in enumerate(nsi):
        if  (W > E) or (Ebi[indx] > E):
            dCS += 0.

        else:
            eps = W*1./E
            epsB = Ebi[indx]*1./E
            dCS += ni * c_pi_efour * eps * (1. + epsB)**(-1.5) * (1. - eps)**(epsB/(epsB+eps)) * ((1. - epsB) +
                                       4. * epsB * log(2.7 + ((1. - eps)/epsB)**(0.5) )/(3.*eps) )   /( W**3)


    return dCS

# 2c) Dielectric function proposed by Powell (1985) for the optical dielectric limit
# see Powell 'Calculations of electron inelastic mean free paths from experimental optical data'
# Surface and and interface analysis, 7(6): 263-274, 1985
#@has_units
# def diel_Pow_dCS(E, W, eps_W, powell_c, c_me=me, c_e=e, c_hbar=hbar):
#     """ Calculate the Powell formula for the dielectric inelastic cross section
#
#         Parameters
#         ----------
#         E      : array : units = eV
#                        incident energy
#
#         W      : array : units = eV
#                        energy loss
#
#         eps_W  : array : units = dim
#                         dielectric function evaulated at W
#
#         Penn_b : array : units = dim
#                         Powell c parameter
#
#         c_me   : array : units = kg
#                        electron mass constant
#
#         c_e    : array : units = coulomb
#                        electron charge constant
#
#         c_hbar : array : units =
#                        reduced Plank's constant
#
#         Returns
#         -------
#         dCS    : array : units = cm**2
#     """
#
#     try:
#         dCS = c_me * c_e**2 * eps_W*ln(powell_c * E/ W) / (2. * pi * c_hbar**2 * E)
#
#         if  (1. - eps < 0):
#             raise E_lossTooLarge
#
#     except E_lossTooLarge:
#         print ' The energy loss is larger than the current electron energy in Powell formulation of the electrostatic discrete CS'
#         print ' W is', W ,'and E is', E
#         print ' Stopping'
#         sys.exit()
#
#     return dCS
