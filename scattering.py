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
            sigma = moller_sigma(self.e.energy, self.free_param['Ec'], self.material.get_nval(), c_pi_efour)
            mfp = mfp_from_sigma(sigma, atnd)
            self.pathl = -mfp * log(self.rn)

        elif(e.type == 'Gryzinski'):
            sigma = gryz_sigma(self.e.energy, self.free_param['Ec'], self.material.get_nval(), c_pi_efour)
            mfp = mfp_from_sigma(sigma, atnd)
            self.pathl = -mfp * log(self.rn)

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
