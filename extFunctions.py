#from scimath.units.api import has_units
from numpy import log
import numpy as np
import numpy.ma as ma
from scipy import stats, integrate
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
    # if ( (type(W) is np.float32) or (type(W) is float) ):
    #     assert (eps<1.0), 'W is larger than E: %s > %s' % (W,E)
    #
    # elif ((type(W) is np.ndarray) or (type(W) is ma.core.MaskedArray)):
    assert (np.all(eps<1.0)), 'W is larger than E'
    # else:
    #     print ('W:', type(W))
    #     sys.exit('W has the wrong type in moller_dCS')


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



#####################################################################
#                     energy loss distributions                     #
#                                                                   #
# NOTE: We don't have analytical expressions for the angular        #
# distribututions for the inelastic scatterings, but we do have the #
# binary collision model for finding a scattering angle based on    #
# the energy loss in the event. We can then look at the energy      #
# loss distributions to understand the theoretical angular distrib. #
#####################################################################
from probTables import maxW_moller, maxW_gryz

#------------------------------- Moller ----------------------------------------
class Moller_W_E(stats.rv_continuous):
    '''
    Energy loss distribution of Moller events
    for a specified incident energy.
    The energy is not a property of the instance,
    instead is set when cdf is called.
    '''
    def __init__(self, nfree, Ef, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Wmin = self.a
        self.Wmax = lambda E : maxW_moller(E, Ef)
        self.nfree = nfree

    def totalInt(self, E):
        tInt, _ = integrate.quad(self.func, self.Wmin, self.Wmax(E), epsabs=1e-39)
        return tInt

    def integral(self, W):
        if (W==self.Wmin):
            WInt=self.func(W)
        else:
            WInt, _ = integrate.quad(self.func, self.Wmin, W, epsabs=1e-39)
        return WInt

    def _cdf(self, W, E):
        # set energy value

        # integrand function at this energy
        self.func = lambda Wvar : moller_dCS(E, Wvar, self.nfree)

        return self.integral(W)/self.totalInt(E)

class Moller_W(stats.rv_continuous):
    '''
    Angular continuous distribution for Moller sattering
    for a distribution of energies
    '''
    def __init__(self, Edist_df, nfree, Ef, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Edist_df = Edist_df
        self.Wmin = self.a
        self.Wmax = self.b
        self.nfree = nfree
        self.Ef = Ef

    def _cdf(self, W):
        ''' cumulative distribution function is just the weighted
        sum of all cmf at different samples of energy'''
        # an instance of Moller_W_E
        Moller_W_dist = Moller_W_E(a=self.a , b=self.b, nfree=self.nfree, Ef=self.Ef)

        # an array of CDFs at the energy values in the distribution
        CDFs = np.array(list(map(Moller_W_dist.cdf, list(W)*len(self.Edist_df.energy.values), self.Edist_df.energy.values)))

        # the cumulative dstribution function for all the E in the DataFrame
        return self.Edist_df.weight.values.dot(CDFs)


#------------------------------- Gryz ------------------------------------------
from crossSections import gryz_sigma

class Gryz_W_E(stats.rv_continuous):
    '''
    Energy loss distribution of Gryzinski events
    for a specified incident energy.
    The energy is not a property of the instance,
    instead is set when cdf is called.
    '''
    def __init__(self, nsi, Ebi, Ef, *args, **kwargs):
        '''
        nsi : dict of number of electron per inner shell
        Esi : dict of binding energies of the inner shells
        '''
        super().__init__(*args, **kwargs)
        self.Wmin = self.a
        self.Wmax = lambda E : maxW_gryz(E, Ef)
        self.nsi = nsi
        self.Ebi = Ebi

    def totalInt(self, E):
        tInt, _ = integrate.quad(self.func, self.Wmin, self.Wmax(E), epsabs=1e-39)
        return tInt

    def integral(self, W):
        if (W==self.Wmin):
            WInt=self.func(W)
        else:
            WInt, _ = integrate.quad(self.func, self.Wmin, W, epsabs=1e-39)
        return WInt

    def _cdf(self, W, E):
        # set energy value

        weighted_sum, sigma_sum = 0, 0
        # weight the contributions of each shell by the cross section
        for shell in self.nsi.keys():
            # integrand function at this energy
            self.func = lambda Wvar : gryz_dCS(E, Wvar, self.nsi[shell], self.Ebi[shell])

            sigma = gryz_sigma(E, self.Ebi[shell], self.nsi[shell])
            weighted_sum += sigma * self.integral(W)/self.totalInt(E)
            sigma_sum += sigma
            
        return weighted_sum/sigma_sum

class Gryz_W(stats.rv_continuous):
    '''
    Angular continuous distribution for Gryzinski sattering
    for a distribution of energies
    '''
    def __init__(self, Edist_df, nfree, Ef, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Edist_df = Edist_df
        self.Wmin = self.a
        self.Wmax = self.b
        self.nsi = nsi
        self.Ebi = Ebi
        self.Ef = Ef

    def _cdf(self, W):
        ''' cumulative distribution function is just the weighted
        sum of all cmf at different samples of energy'''
        # an instance of Gryz_W_E
        Gryz_W_dist = Gryz_W_E(a=self.a , b=self.b, nsi=self.nsi, Ebi=self.Ebi, Ef=self.Ef)

        # an array of CDFs at the energy values in the distribution
        CDFs = np.array(list(map(Gryz_W_dist.cdf, list(W)*len(self.Edist_df.energy.values), self.Edist_df.energy.values)))

        # the cumulative dstribution function for all the E in the DataFrame
        return self.Edist_df.weight.values.dot(CDFs)
