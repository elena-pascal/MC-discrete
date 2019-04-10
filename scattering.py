from math import log, isnan
import numpy as np
from scipy.constants import pi, Avogadro, hbar, m_e, e, epsilon_0, eV

from parameters import u_hbar, u_me, u_e, u_eps0, c_pi_efour
from electron import electron

from scimath.units.api import UnitScalar, UnitArray, convert, has_units
from scimath.units.energy import J, eV, KeV
from scimath.units.electromagnetism import coulomb, farad
from scimath.units.length import m, cm, km, angstrom
from scimath.units.time import s
from scimath.units.mass import g, kg
from scimath.units.density import g_per_cm3, kg_per_m3
from scimath.units.substance import mol
from scimath.units.dimensionless import dim

@has_units
def at_num_dens(dens, atom_mass):
    """ Calculate the atomic number density for given
        density and atomic mass/weight

        Parameters
        ----------
        dens      : array : units = g_per_cm3

        atom_mass : array : units = g/mol

        Returns
        -------
        n         : array : units = m**-3
                   n = dens*A/atom_mass
      """
    A = Avogadro
    n = dens*A/atom_mass * cm**-3/m**-3
    return n


@has_units
def fermi_energy(atNumDens, nvalence, u_hbar, u_me):
    """ Calculate the Fermi energy of a material
        from its density, atomic weight and
        the number of valence electrons

        Parameters
        ----------
        atNumDens : array  : units = m**-3

        u_hbar    : scalar : units = J*s

        u_me      : scalar : units = kg

        Returns
        -------
        Ef        : array : units = eV
                    Ef = hbar**2 * (3.*(pi**2)*n)**(2./3.)/(2.*me)
      """

    n = nvalence * atNumDens

    Ef = u_hbar**2 * (3.*(pi**2)*n)**(2./3.)/(2.*u_me) * m**2*kg*s**-2/ eV
    return Ef


@has_units
def plasmon_energy(atNumDens, nvalence, u_hbar, u_me, u_e, u_eps0):
    """ Calculate the plasmon energy of a material
        from its density, atomic weight and
        the number of valence electrons

        Parameters
        ----------
        atNumDens : array  : units = m**-3

        u_hbar    : scalar : units = J*s

        u_me      : scalar : units = kg

        u_e       : scalar : units = coulomb

        u_eps0    : scalar : units = farad*m**-1

        Returns
        -------
        Epl        : array : units = eV
                    Ef = hbar * ((n * e**2)/(u_eps0 * u_me))**0.5
      """

    n = nvalence * atNumDens

    Epl = u_hbar * e * (n/(u_eps0 * u_me))**0.5  * m**2*kg*s**-2/ eV
    return Epl


@has_units
def mfp_from_sigma(sigma, n):
    """ Calculate the mean free path from the total cross section

        Parameters
        ----------
        sigma  : array : units = cm**2
                total cross section

        n      : array : units = m**-3

        Returns
        -------
        mfp    : array : units = angstrom
    """
    mfp = 1./(n*sigma) * angstrom*cm**2/m**3
    return mfp

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
    dCS_M =  nfree*c_pi_efour* E**2 *( 1./(eps**2) +
                 ( 1./((1.-eps)**2) ) - ( 1./(eps*(1.-eps)) ) )

    return dCS_M


# 2b) Gryzinski differential cross section for core shell electrons
@has_units
def gryz_dCS(E, W, Ebi, nsi, c_pi_efour):
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

    dCS_G = nsi * c_pi_efour * (1. + epsB)**(-1.5) * (1. - eps)**(epsB/(epsB+eps)) * ((1. - epsB) +
                               4. * epsB * np.log(2.7 + ((1. - eps)/epsB)**(0.5) )/3. )   /(eps**2 * E**2)
    return dCS_G



###################################################################
#                       Total stopping power                      #
###################################################################

# 2a) Moller stopping power for free electrons
@has_units
def moller_sp(E, Emin, nfree, n, c_pi_efour):
    """ Calculate the Moller stopping power

        Parameters
        ----------
        E      : array : units = eV

        Emin   : array : units = eV

        nfree  : array : units = dim

        n      : array : units = cm**-3
               atom number density

        c_pi_efour: scalar: units = cm**2 * eV**2

        Returns
        -------
        sp_M   : array : units = eV/angstrom
    """

    eps = Emin*1./E
    sp_M = nfree*n*c_pi_efour*(2. - (1./(1.-eps)) +  np.log( 1./(8.* eps*((1.-eps)**2)) ))/E  * cm**(-1)/angstrom**(-1)
    return sp_M

# 2b) Gryzinski stopping power for core shell electrons
@has_units
def gryz_sp(E, Enl, nsi, n, c_pi_efour):
    """ Calculate the Gryzinski inelastic cross section

        Parameters
        ----------
        E      : array : units = eV

        Enl    : array : units = eV

        nsi    : array : units = dim
               number of electrons in i-shell

        n      : array : units = cm**-3
               atom number density

        c_pi_efour: scalar: units = cm**2 * eV**2

        Returns
        -------
        s_G    : array : units = eV/angstrom
    """

    U = E*1./Enl
    sp_G = nsi * n* c_pi_efour * ((U - 1.)/(U + 1.))**1.5 * (np.log(U) +
                                     4.*np.log(2.7+(U-1.)**0.5)/3.)/E * cm**(-1)/angstrom**(-1)
    return sp_G

# 2c) Quinn stopping power for plasmons
@has_units
def quinn_sp(E, Epl, Ef, n, bohr_r):
    """ Calculate the Quinn inelastic stopping power
        Patrick's formula

        Parameters
        ----------
        E      : array : units = eV
                incident energy

        Epl    : array : units = eV
                plasmon energy

        Ef     : array : units = eV
                Fermi energy

        n      : array : units = m**-3
               atom number density

        bohr_r :scalar : units = m

        Returns
        -------
        sp_Q    : array : units = eV/angstrom
    """
    E_Ef = E*1./Ef
    Epl_Ef = Epl*1./Ef

    s_Q_total = Epl*np.log( ((1. + Epl_Ef)**(0.5) - 1.)/ ( E_Ef**0.5 -
             (E_Ef - Epl_Ef)**0.5 ) )/(2. *  bohr_r * E)
    sp_Q = s_Q_total * m**(-1)/ angstrom**(-1) * Epl

    return sp_Q


# 2d) Bethe continuous stopping power
@has_units
def bethe_cl_sp(Z,E,n,c_pi_efour):
    """ Calculate the Bethe continuous inelastic scattering stopping power
        per atom

        Parameters
        ----------
        Z      : array : units = dim
                atomic number

        E      : array : units = eV
                incident energy

        n      : array : units = cm**-3
               number density

        c_pi_efour: scalar: units = cm**2 * eV**2

        Returns
        -------
        sp_B    : array : units = eV/angstrom
    """
    #the mean ionisation potential J in eV
    if (Z>=13):
        J = (9.76*Z + 58.5/(Z**0.19))
    else:
        J = 11.5*Z

    sp_B = 2.*c_pi_efour*n*(Z/E) * np.log( 1.166*(E )/J ) * cm**(-1)/angstrom**(-1)
    return sp_B



@has_units
def bethe_mod_sp_k(Z,E,n,k,c_pi_efour):
    """ Calculate the Bethe continuous inelastic scattering stopping power
        per atom

        Parameters
        ----------
        Z      : array : units = dim
                atomic number

        E      : array : units = eV
                incident energy

        n      : array : units = cm**-3
               number density

        c_pi_efour: scalar: units = cm**2 * eV**2

        Returns
        -------
        sp_B    : array : units = eV/angstrom
    """
    #the mean ionisation potential J in eV
    if (Z>=13):
        J = (9.76*Z + 58.5/(Z**0.19))
    else:
        J = 11.5*Z

    sp_B = 2.*c_pi_efour*n*(Z/E) * np.log( 1.166*(E + k*J )/J ) * cm**(-1)/angstrom**(-1)
    return sp_B


@has_units
def bethe_mod_sp(E,n, Zi, Ei, Zval, Eval, c_pi_efour):
    """ Calculate the Bethe continuous inelastic scattering stopping power
        per atom

        Parameters
        ----------
        E      : array : units = eV
                incident energy

        n      : array : units = cm**-3
               number density

        Zi     : array : units = dim
                occupancy of shell i

        Ei     : array : units = eV
                binding energy of shell i

        c_pi_efour: scalar: units = cm**2 * eV**2

        Returns
        -------
        sp_B    : array : units = eV/angstrom
    """

    prefactor = (2.*c_pi_efour*n/E ) * cm**(-1)/angstrom**(-1)

    sumi = Zval*np.log(E*1./Eval)
    for indx, Eb in enumerate((Ei)):
        sumi = sumi + Zi[indx]* np.log( E*1./Eb )

    sp_B = prefactor * sumi
    return sp_B




class scatter:
    ''' Scattering can be Rutherford, Browning, Moller, Gryzinski, Quinn or Bethe
    '''
    def __init__(self, type, electron, material, free_param, random_number):
        self.type = type
        self.e = electron
        self.material = material
        self.free_param = free_param
        self.rn = random_number

    def get_Eloss(self):
        if (self.e.type == 'Rutherford'):
            self.e_loss = 0.
            print 'Rutherford elastic scattering has no energy loss'

        if (self.e.type == 'Browning'):
            self.e_loss = 0.
            print 'Browning elastic scattering has no energy loss'

        elif(self.e.type == 'Moller'):

        elif(self.e.type == 'Gryzinski'):

        elif(self.e.type == 'Quinn'):

        elif:
            print 'I did not understand the scattering type in scatter.get_Eloss'


    def get_pathl(self):
        '''
        Path length is calculated from the cross section
        path_length = - mean_free_path * log(rn)
        '''
        atnd = at_num_dens(self.material.get_density(), self.material.get_atwt())

        if (e.type == 'Rutherford'):
            sigma = ruther_sigma(self.e.energy, self.material.get_Z())
            mfp = mfp_from_sigma(sigma, atnd)
            self.pathl = -mfp * log(self.rn)

        elif(e.type == 'Moller'):
            sigma = moller_sigma(self.e.energy, Emin, self.material.get_nval(), c_pi_efour)

        elif(e.type == 'Gryzinski'):

        elif(e.type == 'Quinn'):

        elif:
            print 'I did not understand the scattering type in scatter.get_pathl'


    def get_phi(self):
        if (e.type == 'Rutherford'):
            self.e_loss = 0.
            print 'Rutherford elastic scattering has no energy loss'
        elif(e.type == 'Moller'):

        elif(e.type == 'Gryzinski'):

        elif(e.type == 'Quinn'):

        elif:
            print 'I did not understand the scattering type in scatter.get_phi'
