import numpy as np
from scimath.units.api import UnitScalar, UnitArray, has_units
from scimath.units.energy import  eV
from scimath.units.length import cm, m
from scimath.units.mass import  kg
from scimath.units.time import s

# constants with no units
from scipy.constants import pi, Avogadro, hbar, m_e, e, epsilon_0

#constants with units
from MC.parameters import u_hbar, u_me, u_e, u_eps0

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
    return  dens * Avogadro/atom_mass * m**3/cm**3


@has_units
def fermi_energy(atNumDens, nvalence, c_hbar=hbar, c_me=m_e):
    """ Calculate the Fermi energy of a material from its density,
        atomic weight and the number of valence electrons.

        Call either without passin the constants values for unitless results
        or, alternatively, pass the contants values with units for results with units.

        Parameters
        ----------
        atNumDens : array  : units = m**-3

        c_hbar    : scalar : units = J*s

        c_me      : scalar : units = kg

        Returns
        -------
        Ef        : array : units = eV
                    Ef = hbar**2 * (3.*(pi**2)*n)**(2./3.)/(2.*me)
      """
    n = nvalence * atNumDens

    return c_hbar**2 * (3.*(pi**2)*n)**(2./3.)/(2.*c_me) * m**2*kg*s**-2/ eV



@has_units
def plasmon_energy(atNumDens, nvalence, c_hbar=hbar, c_me=m_e, c_e=e, c_eps0=epsilon_0):
    """ Calculate the plasmon energy of a material from its density,
        atomic weight and the number of valence electrons.

        Call either without passin the constants values for unitless results
        or, alternatively, pass the contants values with units for results with units.

        Parameters
        ----------
        atNumDens : array  : units = m**-3

        c_hbar    : scalar : units = J*s

        c_me      : scalar : units = kg

        c_e       : scalar : units = coulomb

        c_eps0    : scalar : units = farad*m**-1

        Returns
        -------
        Epl        : array : units = eV
                    Ef = hbar * ((n * e**2)/(c_eps0 * c_me))**0.5
      """

    n = nvalence * atNumDens

    return c_hbar * e * (n/(c_eps0 * c_me))**0.5  * m**2*kg*s**-2/ eV



@has_units
def powell_c(penn_b, plasmon_E):
    """ Powell's c function in terms of Penn's b parameter
        used in the derivation of IMFP using the dielectric function

        Parameters
        ----------
        penn_b     : array : units = dim
                  Penn's b material parameter

        plasmon_E  : array : units = eV
                  plasmon energy

        Returns
        -------
        c          : array : units = dim
                    c = plasmon_E * exp(penn_b)

    """
    return plasmon_E * np.exp(penn_b)/eV





class material:
    ''' material is a class containing all the necessary
        material parameters for a small predified subset of species
    '''
    def __init__(self, species):
        self.species = species
        self.params = scatter_params(species)

        # atomic number density
        self.atnd =  at_num_dens(self.params['density'], self.params['atwt'])
        # plasmon energy
        self.plasmon_e = plasmon_energy(self.atnd, self.params['n_val'])
        # fermi energy
        self.fermi_e = fermi_energy(self.atnd, self.params['n_val'])

        # Powell c parameter
        self.powell_c = powell_c(self.params['penn_b'], self.plasmon_e)

    def set_units(self):
        ''' in the case we want units'''
        self.params.update({'n_val': UnitArray(self.params['n_val'], units="dim") })
        self.params.update({'E_val': UnitScalar(self.params['E_val'], units="eV") })
        self.params.update({'Es': UnitArray(self.params['Es'], units="eV") })
        self.params.update({'density': UnitScalar(self.params['density'], units="g_per_cm3") })
        self.params.update({'atwt': UnitScalar(self.params['atwt'], units="g/mol") })

        self.atnd = at_num_dens(self.params['density'], self.params['atwt'])
        self.plasmon_e = plasmon_energy(self.atnd, self.params['n_val'], u_hbar, u_me, u_e, u_eps0)
        self.fermi_e = fermi_energy(self.atnd, self.params['n_val'], u_hbar, u_me)




def scatter_params(species):
    if species=='Al':
        material = {'species': 'Al'}

        # number of valence electrons
        material['n_val'] = 3

        # energy of valence shell
        material['E_val'] = 72.55

        # name of valence shells
        material['name_val'] = '3s2-3p1'

        # energy levels for core electrons
        material['name_s'] = ['1s', '2s', '2p']

        # binding energies
        material['Es'] = {'1s':1559., '2s':118., '2p':73.5}

        # number of electrons per shell
        material['ns'] = {'1s':2, '2s':2, '2p':6}

        # atomic  number
        material['Z'] = 13

        #material density
        material['density'] = 2.70

        # atomic weight
        material['atwt'] = 26.98154

        # modified Bethe k value from Joy and Luo, Scanning 1989
        material['bethe_k'] = 0.815

        # Penn's b value for IMFP from Penn, Journal of Electron Spectroscopy and Related Phenomena, 9, 1976
        material['penn_b'] = -2.16


    elif species=='Si':
        material = {'species': 'Si'}

        # number of valence electrons
        material['n_val'] = 4

        # energy of valence shell
        material['E_val'] = 99.42

        # name of valence shells
        material['name_val'] = '3s2-3p2'

        # energy levels for core electrons
        material['name_s'] = ['1s', '2s', '2p']

        # binding energies
        material['Es'] = {'1s':1189, '2s':149, '2p':100}

        # number of electrons per shell
        material['ns'] = {'1s':2, '2s':2, '2p':6}

        # atomic  number
        material['Z'] = 14

        #material density
        material['density'] = 2.33

        # atomic weight
        material['atwt'] = 28.085

        # modified Bethe k value from Joy and Luo, Scanning 1989
        material['bethe_k'] = 0.822

        # Penn's b value for IMFP from Penn, Journal of Electron Spectroscopy and Related Phenomena, 9, 1976
        material['penn_b'] = -2.19

        # extinction distance from EMqg for h,k,l = 0,0,4 in A at 20keV
        material['xip_g'] = 1360


    elif species=='Cu':
        material = {'species': 'Cu'}

        # number of valence electrons
        material['n_val'] = 11

        # energy of valence shell
        material['E_val'] = 75.1

        # name of valence shells
        material['name_val'] = '3d10-4s2'

        # energy levels for core electrons
        material['name_s'] = ['1s', '2s2p', '3s', '3p']

        # binding energies
        material['Es'] = {'1s':8980, '2s2p':977, '3s':120, '3p':74}

        # number of electrons per shell
        material['ns'] = {'1s':2, '2s2p':8, '3s':2, '3p':6}

        # atomic  number
        material['Z'] = 29

        #material density
        material['density'] = 8.96

        # atomic weight
        material['atwt'] = 63.546

        # modified Bethe k value from Joy and Luo, Scanning 1989
        material['bethe_k'] = 0.83

        # Penn's b value for IMFP from Penn, Journal of Electron Spectroscopy and Related Phenomena, 9, 1976
        material['penn_b'] = -3.21


    elif species=='Au':
        material = {'species': 'Au'}

        # number of valence electrons
        material['n_val'] = 11

        # energy of valence shell
        material['E_val'] = 75.1

        # name of valence shells
        material['name_val'] = '5d10-6s1'

        # energy levels for core electrons
        material['name_s'] = ['2s2p', '3s3p3d', '4s4p', '4d', '5s', '4f', '5p2', '5p4']

        # binding energies
        material['Es'] = {'2s2p':12980, '3s3p3d':2584, '4s4p':624, '4d':341, '5s':178, '4f':85, '5p2':72, '5p4':54}

        # number of electrons per shell
        material['ns'] = {'2s2p':8, '3s3p3d':18, '4s4p':8, '4d':10, '5s':2, '4f':14, '5p2':2, '5p4':4}

        # atomic  number
        material['Z'] = 79

        #material density
        material['density'] = 19.30

        # atomic weight
        material['atwt'] = 196.967

        # modified Bethe k value from Joy and Luo, Scanning 1989
        material['bethe_k'] = 0.851

        # Penn's b value for IMFP from Penn, Journal of Electron Spectroscopy and Related Phenomena, 9, 1976
        material['penn_b'] = -2.16

    return material
