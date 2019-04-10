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



class material:
    ''' material is a class containing all the necessary
        material parameters for a small predified subset of species
    '''
    def __init__(self, species):
        self.species = species
        self.params = scattering_params(species)

    def get_nval(self):
        ''' number of valence electrons '''
        return self.params['n_val']

    def get_Eval(self):
        ''' energy of valence shell '''
        return self.params['E_val']

    def get_name_s(self):
        ''' array of names of core shells '''
        return self.params['name_s']

    def get_Es(self):
        ''' array of energies of core shells as UnitArray'''
        return self.params['Ei']

    def get_ns(self):
        ''' array with number of electrons in core shells'''
        return self.params['ns']

    def get_Z(self):
        ''' atomic  number'''
        return self.params['Z']

    def get_density(self):
        ''' material density as UnitScalar'''
        return self.params['density']

    def get_atwt(self):
        ''' atomic weight as UnitScalar'''
        return self.params['atwt']

    def get_Bethe_k(self):
        ''' modified Bethe k value from Joy and Luo, Scanning 1989'''
        return self.params['k']




def scattering_params(species):
    if species=='Al':
        material = {'species': 'Al'}
        # number of valence electrons
        material['n_val'] = 3

        material['E_val'] = UnitScalar(72.55, units="eV")

        # energy levels for core electrons
        material['name_s'] = ['1s', '2s', '2p']
        # binding energies
        material['Es'] = UnitArray((1559, 118, 73.5), units="eV")
        # number of electrons per shell
        material['ns'] = np.array([2, 2, 6])

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
