import numpy as np
from scimath.units.api import UnitScalar, UnitArray, convert, has_units
from scimath.units.energy import J, eV, KeV
from scimath.units.electromagnetism import coulomb, farad
from scimath.units.length import m, cm, km, angstrom
from scimath.units.time import s
from scimath.units.mass import g, kg
from scimath.units.density import g_per_cm3, kg_per_m3
from scimath.units.substance import mol
from scimath.units.dimensionless import dim

from scipy.constants import pi, Avogadro, hbar, m_e, e, epsilon_0
from parameters import u_hbar, u_me, u_e, u_eps0, c_pi_efour

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
    n = dens*A/atom_mass * cm**(-3)/m**-3
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


class material:
    ''' material is a class containing all the necessary
        material parameters for a small predified subset of species
    '''
    def __init__(self, species):
        self.species = species
        self.params = scattering_params(species)

    def nval(self):
        ''' number of valence electrons '''
        return self.params['n_val']

    def Eval(self):
        ''' energy of valence shell '''
        return self.params['E_val']

    def name_s(self):
        ''' array of names of core shells '''
        return self.params['name_s']

    def Es(self):
        ''' array of energies of core shells as UnitArray'''
        return self.params['Es']

    def ns(self):
        ''' array with number of electrons in core shells'''
        return self.params['ns']

    def Z(self):
        ''' atomic  number'''
        return self.params['Z']

    def density(self):
        ''' material density as UnitScalar'''
        return self.params['density']

    def atwt(self):
        ''' atomic weight as UnitScalar'''
        return self.params['atwt']

    def Bethe_k(self):
        ''' modified Bethe k value from Joy and Luo, Scanning 1989'''
        return self.params['k']

    # some very useful parameters
    def atnd(self):
        ''' atomic number density'''
        return at_num_dens(self.params['density'], self.params['atwt'])

    def pl_e(self):
        ''' plasmon energy'''
        return plasmon_energy(self.atnd(), self.params['n_val'], u_hbar, u_me, u_e, u_eps0)

    def fermi_e(self):
        ''' fermi energy'''
        return fermi_energy(self.atnd(), self.params['n_val'], u_hbar, u_me)



def scattering_params(species):
    if species=='Al':
        material = {'species': 'Al'}
        # number of valence electrons
        material['n_val'] = UnitArray((3), units="dim")

        material['E_val'] = UnitScalar(72.55, units="eV")

        # energy levels for core electrons
        material['name_s'] = ['1s', '2s', '2p']
        # binding energies
        material['Es'] = UnitArray((1559, 118, 73.5), units="eV")
        # number of electrons per shell
        material['ns'] = UnitArray((2, 2, 6), units="dim")

        material['Z'] = 13
        material['density'] = UnitScalar(2.70, units="g_per_cm3")
        material['atwt'] = UnitScalar(26.98154, units="g/mol")

        # modified Bethe k value from Joy and Luo, Scanning 1989
        material['k'] = 0.815

    return material



# ##### Scattering variables for Si
# # number of valence electrons
# n_val_Si = 4
#
# E_val_Si = UnitScalar(99.42, units="eV")
#
# # energy levels for core electrons
# name_s_Si = ['1s', '2s', '2p']
# # binding energies
# Ei_Si = UnitArray((1189, 149, 100), units="eV")
# # number of electrons per shell
# ni_Si = np.array([2, 2, 6])
#
# Z_Si = 14
# density_Si = UnitScalar(2.33, units="g_per_cm3")
# atwt_Si = UnitScalar(28.085, units="g/mol")
#
# # modified Bethe k value from Joy and Luo, Scanning 1989
# k_Si = 0.822
#
# # incident energy
# E_Si = UnitScalar(20000, units="eV")
# # 5..100
# Ec_Si = UnitScalar(10., units="eV")
#
#
# #### Scattering variables for Cu
# # number of valence electrons
# n_val_Cu = 11
#
# E_val_Cu = UnitScalar(75.1, units="eV")
#
# # energy levels for core electrons
# name_s_Cu = ['1s', '2s2p', '3s', '3p']
# # binding energies
# Ei_Cu = UnitArray((8980, 977, 120, 74), units="eV")
# # number of electrons per shell
# ni_Cu = np.array([2, 8, 2, 6])
#
# Z_Cu = 29
# density_Cu = UnitScalar(8.96, units="g_per_cm3")
# atwt_Cu = UnitScalar(63.546, units="g/mol")
#
# # modified Bethe k value from Joy and Luo, Scanning 1989
# k_Cu = 0.83
#
# # incident energy
# E_Cu = UnitScalar(20000, units="eV")
# # 5..100
# Ec_Cu = UnitScalar(10., units="eV")
#
#
#
# #### Scattering variables for Au
# # number of valence electrons
# n_val_Au = 11
#
# # energy levels for core electrons
# name_s_Au = ['2s2p', '3s3p3d', '4s4p', '4d', '5s', '4f', '5p2', '5p4']
# # binding energies
# Ei_Au = UnitArray((12980, 2584, 624, 341, 178, 85, 72, 54), units="eV")
# # number of electrons per shell
# ni_Au = np.array([8, 18, 8, 10, 2, 14, 2, 4])
#
# Z_Au = 79
# density_Au = UnitScalar(19.30, units="g_per_cm3")
# atwt_Au = UnitScalar(196.967, units="g/mol")
#
# # modified Bethe k value from Joy and Luo, Scanning 1989
# k_Au = 0.851
#
# # incident energy
# E_Au = UnitScalar(20000, units="eV")
# # 5..100
# Ec_Au = UnitScalar(10., units="eV")
