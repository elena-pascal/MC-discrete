import numpy as np
from scipy.constants import pi
from scipy import stats

from scimath.units.api import has_units
from scimath.units.length import cm, m
from scimath.units.energy import eV, KeV

from MC.parameters import pi_efour, bohr_r



#@has_units
def alpha(energy, Z):
    ''' screening parameter used in Rutherford scattering

        Parameters
        ----------
        energy : array : units = eV

        Z      : array : units = dim

        Returns
        -------
        alpha  : array : units = dim
    '''
    return  3.4*(Z**(0.666667)) / energy

###################################################################
#                                                                 #
#                      Elastic total cross section                #
#                                                                 #
###################################################################

#### 1a) Elastic Rutherford scattering cross section
#@has_units
def ruther_sigma(E, Z):
    """ Calculate the Rutherford elastic cross section
        per atom

        Parameters
        ----------
        E      : array : units = eV

        Z      : array : units = dim

        Returns
        -------
        s_R    : array : units = cm**2
    """
    # compute the screening factor
    alphaR = alpha(E, Z)

    # E must be in keV in Joy's formula
    E = E * eV/KeV

    # correction factor for angular deflection of inelastic scattering
    # Z**2 - > Z*(Z + 1)

    s_R = 5.21e-21 * (Z*Z/(E*E))  * ((E + 511.)/(E + 1024.))**2 * (4.*pi)/(alphaR*(1. + alphaR))
    return s_R


#### 1a') Elastic Rutherford scattering cross section but corrected for inelastic deflections
@has_units
def ruther_sigma_wDefl(E, Z):
    """ Calculate the Rutherford elastic cross section
        per atom

        Parameters
        ----------
        E      : array : units = eV

        Z      : array : units = dim

        Returns
        -------
        s_R    : array : units = cm**2
    """
    E = E * eV/KeV

    # correction factor for angular deflection of inelastic scattering
    # Z**2 - > Z*(Z + 1)

    # compute the screening factor
    alphaR = alpha(E, Z)

    s_R = 5.21e-21 * (Z*(Z+1)/(E**2)) * (4.*pi)/(alpha*(1. + alpha)) * ((E + 511.)/(E + 1024.))**2
    return s_R


#### 1a") Elastic Rutherford scattering cross section with Nigraru screening parameter
@has_units
def ruther_N_sigma(E, Z, c_pi_efour=pi_efour):
    """ Calculate the Rutherford elastic cross section
        with Nigram parameter
        per atom

        Parameters
        ----------
        E      : array : units = eV

        Z      : array : units = dim

        c_pi_efour: scalar: units = cm**2 * eV**2

        Returns
        -------
        s_R    : array : units = cm**2
    """

    beta_N = 5.43 * Z**(2/3)/E

    # this value has been tweaked by fitting Rutherford CS to the pwem model
    # see Adesida, Schimizu, Everhart (1980) JAP 51 (11)
    beta_N_star = 0.48*beta_N

    s_R = c_pi_efour * Z**2/ (4*beta_N_star*(1+beta_N_star)*E**2)

    # correction factor for angular deflection of inelastic scattering
    # Z**2 - > Z*(Z + 1)
    # s_R = c_pi_efour * (Z+1)*Z/ (4*beta_N_star*(1+beta_N_star)*E**2)
    return s_R

### 1a"') Elastic Rutherford scattering cross section with Nigraru screening parameter and Z+1
@has_units
def ruther_N_sigma_wDefl(E, Z, c_pi_efour=pi_efour):
    """ Calculate the Rutherford elastic cross section
        per atom

        Parameters
        ----------
        E      : array : units = eV

        Z      : array : units = dim

        c_pi_efour: scalar: units = cm**2 * eV**2

        Returns
        -------
        s_R    : array : units = cm**2
    """

    beta_N = 5.43 * Z**(2/3)/E

    # this value has been tweaked by fitting Rutherford CS to the pwem model
    # see Adesida, Schimizu, Everhart (1980) JAP 51 (11)
    beta_N_star = 0.48*beta_N

    # correction factor for angular deflection of inelastic scattering
    # Z**2 - > Z*(Z + 1)
    s_R = c_pi_efour * (Z+1)*Z/ (4*beta_N_star*(1+beta_N_star)*E**2)
    return s_R

### 1b) Browning
@has_units
def browning_sigma(E, Z, c_pi_efour=pi_efour):
    """ Calculate the Browning (1992) elastic cross section
        per atom

        Parameters
        ----------
        E      : array : units = eV

        Z      : array : units = dim

        c_pi_efour: scalar: units = cm**2 * eV**2

        Returns
        -------
        s_R    : array : units = cm**2
    """
    Z133 = Z**1.33
    u = np.log10(8 * E * Z133**(-1))
    s_B = 4.7 * 10**(-18) * (Z133 + 0.032*Z*Z)/(E + (0.0155*Z133*E**0.5) )/(1 - 0.02*Z**0.5*np.exp(-u**2))
    return s_B



###################################################################
#                                                                 #
#                      Inelastic total cross section              #
#                                                                 #
###################################################################


#### 2a) Inelastic Moller cross section for free electrons
@has_units
def moller_sigma(E, Emin, nfree, c_pi_efour=pi_efour):
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
    # if the energy of this electron is lower than the cut off energy for free
    # electron scattering then Moller sigma is set to zero
    if (E <= Emin):
        s_M = 0.
    else:
        eps = Emin*1./E
        s_M = nfree*c_pi_efour*(1./(E*E)) * (1./eps - 1./(1.-eps) + np.log(eps/(1.-eps)))
    return s_M


#### 2b) Inelastic Gryzinski cross section for core shell electrons
@has_units
def gryz_sigma(E, Esi, nsi, c_pi_efour=pi_efour):
    """ Calculate the Gryzinski inelastic cross section
        per atom

        Parameters
        ----------
        E      : scalar : units = eV
                primary electron energy

        Esi    : scalar : units = eV
                shell i binding energy

        nsi    : scalar : units = dim
                 number of electrons in i-shell

        c_pi_efour: scalar: units = cm**2 * eV**2

        Returns
        -------
        s_G    : array : units = cm**2
    """
    # if the energy of this electron is lower than the shell i binding energy
    # then this Gryzinski sigma is set to zero
    if (E <= Esi):
        s_G = 0.
    else:
        U = E*1./Esi
        s_G = nsi * c_pi_efour * ((U - 1.)/(U + 1.))**1.5 * (1. + (2.*(1.-(1./(2.*U)))/3. *\
              np.log(2.7 + ((U - 1.)**0.5))) ) /(E*Esi)
    return s_G


#### 2c) Inelastic Quinn cross section for plasmons
@has_units
def quinn_sigma(E, Epl, Ef, n, c_bohr_r=bohr_r):
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

        c_bohr_r : scalar: units = m

        Returns
        -------
        s_Q    : array : units = cm**2
    """

    if (E <= Epl):
        s_Q = 0.
    else:
        E_Ef = E*1./Ef
        Epl_Ef = Epl*1./Ef

        s_Q_total = Epl * np.log( ((1. + Epl_Ef)**(0.5) - 1.)/ ( E_Ef**0.5 -\
             (E_Ef - Epl_Ef)**0.5 ) )/(2. *  c_bohr_r * E)

    s_Q = (s_Q_total/n) * m**2/ cm**2


    return s_Q


#### 2d) Inelastic cross section using the dielectric function
@has_units
def diel_sigma(E, ELF, powell_c, n):
    """ Calculate the dielectric funtion cross section in the optical limit with Powell
        per atom

        Parameters
        ----------
        E        : array : units = eV
                 incident energy

        ELF      : array : units = dim
                 energy loss function = Im(-1/eps(W))

        powell_c : array : units = dim
                 Powells constant

        n        : array : units = m**-3
                 number density

        Returns
        -------
        s_Q    : array : units = cm**2
    """

    function = ELF(W)* ln(powell_c*E/W)
    intergral = trapez(function, 0., E)

    return integral/(3.325*E*n)



###################################################################
#                                                                 #
#                      Elastic differential cross section         #
#                                                                 #
###################################################################

@has_units
def ruther_sigma_p(E, Z, theta):
    """ Calculate the Rutherford elastic differential cross section
        per atom

        Parameters
        ----------
        E      : array : units = eV

        Z      : array : units = dim

        theta  : array : units = dim
                 scattering angle in degrees

        Returns
        -------
        s_p_R    : array : units = cm**2
    """
    E = E * eV/KeV

    # correction factor for angular deflection of inelastic scattering
    # Z**2 - > Z*(Z + 1)

    # compute the screening factor
    alphaR = alpha(E, Z)

    s_p_R = 5.21e-21 * (Z*Z/(E*E)) * ((E + 511.)/(E + 1024.))**2 /(((np.sin(np.radians(theta)*0.5))**2 + alphaR)**2)
    return s_p_R


#####################################################################
#               differential cross section distributions            #
#####################################################################
class Ruth_diffCS_E(stats.rv_continuous):
    '''
    Angular continuous distribution for Rutherford sattering
    at a specific energy
    '''
    def __init__(self, Z, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Z = Z

    def _pdf(self, theta, E):
        ''' probability distribution function'''
        return np.array(ruther_sigma_p(E, self.Z, theta))

    def _cdf(self, theta, E):
        ''' cumulative distribution function'''
        alphaR = np.array(alpha(E, self.Z))
        return (-alphaR-1)*(1-np.cos(np.radians(theta)))/(-2*alphaR + np.cos(np.radians(theta))-1)

    def _ppf(self, x, E):
        '''
        quantile function which applied to a set of random variants from a
        uniform distribution returns a set of random variants from
        the Rutherford angular scattering distribution.

        The Rutherford scattering is a continuous distribution and its CDF can be
        inversed analytically.

        When rvs is called this is the function used if present
        '''
        alphaR = np.array(alpha(E, self.Z))
        return np.degrees(np.arccos(1 - (2*alphaR*x)/(1+alphaR -x)))



class Ruth_diffCS(stats.rv_continuous):
    '''
    Angular continuous distribution for Rutherford sattering
    for a distribution of energies
    '''
    def __init__(self, Edist_df, Z, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Edist_df = Edist_df
        self.Z = Z

    def _cdf(self, theta):
        ''' cumulative distribution function is just the weighted
        sum of all cmf at different samples of energy'''
        # an instance of Ruth_diffCS
        R_dist = Ruth_diffCS_E(a=self.a, b=self.b, Z=self.Z)

        # the cumulative dstribution function for all the E in the DataFrame
        return self.Edist_df.weight.values.dot(R_dist.cdf(theta, self.Edist_df.energy.values))
