from math import log, isnan
import numpy as np
import random

import operator
import collections # sweet sweet ordered dictionaries
from scipy.constants import pi, Avogadro, hbar, m_e, e, epsilon_0, eV
from scipy.interpolate import Rbf

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

#####################################################################
####################### Travel class ###############################
#####################################################################

class travel:
    ''' Let the electron travel a path lenght untill
       an elastic scattering event or a discrete inelastic scattering
       event occurs. Type of travelling can be CSDA or discrete
    '''
    def __init__(self, model, electron, material):
        self.e = electron
        self.type = model
        self.material = material
        self.pathl = pathl

    pl_e = plasmon_energy(atnd, material.get_nval(), u_hbar, u_me, u_e, u_eps0)
    f_e = fermi_energy(atnd, material.get_nval(), u_hbar, u_me)
    atnd = at_num_dens(self.material.get_density(), self.material.get_atwt())

    def compute_pathl(self):
        '''
        Path length is calculated from the cross section
        path_length = - mean_free_path * log(rn)
        '''

        sigma_R = ruther_sigma(self.e.energy, self.material.get_Z())
        mfp_R = mfp_from_sigma(sigma_R, atnd)

        sigma_M = moller_sigma(self.e.energy, self.free_param['Ec'], self.material.get_nval(), c_pi_efour)
        mfp_M = mfp_from_sigma(sigma_M, atnd)

        sigma_G = gryz_sigma(self.e.energy, self.material.get_Es(), self.material.get_ns(), c_pi_efour)
        mfp_G = mfp_from_sigma(sigma_G, atnd)

        sigma_Q = quinn_sigma(self.e.energy, pl_e, f_e, atnd, bohr_r)
        mfp_Q = mfp_from_sigma(sigma_Q, atnd)
        self.pathl = -mfp * log(random.random())

    elif(self.type == 'Bethe'):
    if (self.pathl == 0.):
        print "I'm not telling you how to live your life, but it helps to calculate path lengths before energy losses for CSDA"
    else:
        self.E_loss = self.path * u2n(bethe_mod_sp(self.e.energy, atnd, self.material.get_ns(), \
                                        self.material.get_Es(), self.material.get_Zval(), self.material.get_Eval(), c_pi_efour))

#####################################################################
####################### Scatter class ###############################
#####################################################################
class scatter:
    ''' Scattering can be Rutherford, Browning, Moller, Gryzinski, Quinn or Bethe
    '''
    def __init__(self, electron, material, *free_param, *tables_EW):
        self.e = electron
        self.material = material
        self.free_param = free_param

        if ((self.type == 'Moller') or (self.type == 'Gryzinski')):
            # Moller tables are [Ei, Wi, Int(Ei, Wi)]
            # Gryz tables are [ni, Ei, Wi, Int(Ei, Wi)]
            self.tables_EW = tables_EW

        # intitalise
        self.sigma = {} # dictionary keeping all sigmas
        self.mfp = {} # dictionary keeping all mfp

        self.pathl = 0.
        self.type = 'type'
        self.E_loss = 0.
        self.c2_halfPhi = 0.
        self.halfTheta = 0.


    # some very useful parameters
    pl_e = plasmon_energy(atnd, material.get_nval(), u_hbar, u_me, u_e, u_eps0)
    f_e = fermi_energy(atnd, material.get_nval(), u_hbar, u_me)
    atnd = at_num_dens(self.material.get_density(), self.material.get_atwt())

    sigma['Rutherford'] = ruther_sigma(self.e.energy, self.material.get_Z())
    mfp['Rutherford'] = mfp_from_sigma(sigma['Rutherford'], atnd)

    sigma['Moller'] = moller_sigma(self.e.energy, self.free_param['Ec'], self.material.get_nval(), c_pi_efour)
    mfp['Moller'] = mfp_from_sigma(sigma['Moller'], atnd)

    for i in len(self.material.get_Es()):
        sigma['Gryzinski' + name_s[i]] = gryz_sigma(self.e.energy, self.material.get_Es()[i], self.material.get_ns()[i], c_pi_efour)
        mfp['Gryzinski' + name_s[i]] = mfp_from_sigma(sigma['Gryzinski' + name_s[i]], atnd)

    sigma['Quinn'] = quinn_sigma(self.e.energy, pl_e, f_e, atnd, bohr_r)
    mfp['Quinn'] = mfp_from_sigma(sigma['Quinn'], atnd)

    sigma_total = sum(sigma.values())
    mfp_total = 1. /sum(1./mfp.values())

    def compute_pathl(self):
        '''
        Path length is calculated from the cross section
        path_length = - mean_free_path * log(rn)
        '''

        self.pathl = -mfp_total * log(random.random())

    def det_type(self):
        # bisect for a random number a sorted instance of a dictionary containing
        # scattering cumulative probabilities. if R <= cum.prob.scatter.type - > scatter is of type type
        sorted_sigma = OrderedDict(sorted(sigma.items(), key=operator.itemgetter(1)))
        scatProb = {x: np.cumsum(sorted_sigma.values()[::-1])/sigma_total for x ordered_sigma}

        prob = bisect.bisect_left(scatProb.values(), random.random(()))

        self.type = scatProb.keys()[mydict.values().index(prob)]

    def compute_Eloss(self):
        '''
        energy loss is calculated from tables for the Moller and Gryz type
        '''
        if (self.type == 'Rutherford'):
            self.E_loss = 0.
            print 'Rutherford elastic scattering has no energy loss'

        elif (self.type == 'Browning'):
            self.E_loss = 0.
            print 'Browning elastic scattering has no energy loss'



        elif(self.type == 'Moller'):
            # get the limits of W integration
            # a, b = u2n(extF_limits_moller(self.e.energy, self.free_param['Ec']))
            # populate integral table[Ei, Wi]
            # tables_EW = trapez_table(a, b, self.free_param['Ec'], self.e.energy, self.material.get_nval(), \
            #                            moller_dCS, 100, 1000, self.free_param['Ec'])


            #wi = np.linear(a, b, 100)
            #ei = np.linear( self.free_param['Ec'], self.e.energy, 1000)
            # rbfi = Rbf(tables_EW[0], tables_EW[1], tables_EW[2])  # radial basis function interpolator instance

            # integral(E, Wi) is rr * total integral
            integral = random.random() * tables_EW[2][self.e.energy, -1]
            return bisect.bisect_left(tables_EW[1][:, 1], integral)


        elif(self.type == 'Gryzinski'):
            for i, ishell in enumerate(self.material.get_ns()):
                integral = random.random() * tables_EW[2][self.e.energy, -1]
                return bisect.bisect_left(tables_EW[1][:, 1], integral)


        elif(self.type == 'Quinn'):
            self.E_loss = pl_e

        elif:
            print 'I did not understand the type of scattering in scatter.calculate_Eloss'


    def calculate_sAngles(self):
        if (self.type == 'Rutherford'):
            alpha =  3.4e-3*(self.material.get_Z()**(0.67))/self.e.energy
            self.c2_halfPhi = 1. - alpha*random.random()/(1.+alpha-random.random())
            self.halfTheta = pi*random.random

        elif(self.type == 'Bethe'):

            self.halfTheta = pi*random.random

        elif((self.type == 'Moller') OR (e.type == 'Gryzinski')):
            if (self.E_loss == 0.):
                print "I'm not telling you how to live your life, but it helps to calculate the lost energy before the scattering angles for inelastic events"
            else:
                self.c2_halfPhi = 0.5*((1.-(self.Eloss/self.e.energy))**0.5 + 1)
                self.halfTheta = pi*random.random

        elif:
            print 'I did not understand the scattering type in scatter.calculate_sAngles'
