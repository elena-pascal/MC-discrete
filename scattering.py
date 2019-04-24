from math import log, isnan
import numpy as np
import random
import bisect

import operator
from collections import OrderedDict # sweet sweet ordered dictionaries
from scipy.constants import pi, Avogadro, hbar, m_e, e, epsilon_0, eV
from scipy.interpolate import Rbf

from parameters import u_hbar, u_me, u_e, u_eps0, c_pi_efour, u_bohrr
from electron import electron
from crossSections import ruther_sigma, moller_sigma, gryz_sigma, quinn_sigma

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

# #####################################################################
# ####################### Travel class ###############################
# #####################################################################
#
# class travel:
#     ''' Let the electron travel a path lenght untill
#        an elastic scattering event or a discrete inelastic scattering
#        event occurs. Type of travelling can be CSDA or discrete
#     '''
#     def __init__(self, model, electron, material):
#         self.e = electron
#         self.type = model
#         self.material = material
#         self.pathl = pathl
#
#     pl_e = plasmon_energy(atnd, material.get_nval(), u_hbar, u_me, u_e, u_eps0)
#     f_e = fermi_energy(atnd, material.get_nval(), u_hbar, u_me)
#     atnd = at_num_dens(self.material.get_density(), self.material.get_atwt())
#
#     def compute_pathl(self):
#         '''
#         Path length is calculated from the cross section
#         path_length = - mean_free_path * log(rn)
#         '''
#
#         sigma_R = ruther_sigma(self.e.energy, self.material.get_Z())
#         mfp_R = mfp_from_sigma(sigma_R, atnd)
#
#         sigma_M = moller_sigma(self.e.energy, self.free_param['Ec'], self.material.get_nval(), c_pi_efour)
#         mfp_M = mfp_from_sigma(sigma_M, atnd)
#
#         sigma_G = gryz_sigma(self.e.energy, self.material.get_Es(), self.material.get_ns(), c_pi_efour)
#         mfp_G = mfp_from_sigma(sigma_G, atnd)
#
#         sigma_Q = quinn_sigma(self.e.energy, pl_e, f_e, atnd, bohr_r)
#         mfp_Q = mfp_from_sigma(sigma_Q, atnd)
#         self.pathl = -mfp * log(random.random())


#####################################################################
####################### Scatter class ###############################
#####################################################################
class scatter:
    ''' Scattering can be Rutherford, Browning, Moller, Gryzinski, Quinn or Bethe
    '''
    def __init__(self, electron, material, free_param, tables_EW_M, tables_EW_G):
        self.e = electron
        self.material = material
        self.free_param = free_param

        self.pathl = 0.
        self.type = None
        self.E_loss = 0.
        self.c2_halfPhi = 0.
        self.halfTheta = 0.


        # some very useful parameters
        atnd = material.get_atnd()
        pl_e = material.get_pl_e()
        f_e = material.get_fermi_e()

        # intitalise
        self.sigma = {} # dictionary keeping all sigmas
        self.mfp = {} # dictionary keeping all mfp

        self.sigma['Rutherford'] = ruther_sigma(electron.energy, material.get_Z())
        self.mfp['Rutherford'] = mfp_from_sigma(self.sigma['Rutherford'], atnd)

        self.sigma['Moller'] = moller_sigma(electron.energy, free_param, material.get_nval(), c_pi_efour)
        self.mfp['Moller'] = mfp_from_sigma(self.sigma['Moller'], atnd)

        for i in range(len(material.get_Es())):
            self.sigma['Gryzinski' + material.get_name_s()[i]] = gryz_sigma(electron.energy, material.get_Es()[i], material.get_ns()[i], c_pi_efour)
            self.mfp['Gryzinski' + material.get_name_s()[i]] = mfp_from_sigma(self.sigma['Gryzinski' + material.get_name_s()[i]], atnd)

        self.sigma['Quinn'] = quinn_sigma(electron.energy, pl_e, f_e, atnd, u_bohrr)
        self.mfp['Quinn'] = mfp_from_sigma(self.sigma['Quinn'], atnd)

        self.sigma_total = sum(self.sigma.values())
        self.mfp_total = 1. /sum(1./np.array(self.mfp.values()))


    def compute_pathl(self):
        '''
        Path length is calculated from the cross section
        path_length = - mean_free_path * log(rn)
        '''

        self.pathl = -self.mfp_total * log(random.random())

    def det_type(self):
        '''
        bisect for a random number a sorted instance of a dictionary containing
        scattering cumulative probabilities. if R <= cum.prob.scatter.type - > scatter is of type type
        Set the type of scattering after determining type
        '''
        # ordered dicitonary of sigmas
        sorted_sigmas = OrderedDict(sorted(self.sigma.items(), key=operator.itemgetter(1), reverse=True))
        # probabilities to compare the random number against are cumulative sums of the
        # reversed ordered dictionary
        print 'sorted dict',sorted_sigmas
        probs = np.cumsum(sorted_sigmas.values())/np.array(self.sigma_total)

        print 'here', [probs[i] for i, (k,v) in enumerate(sorted_sigmas.items())]
        scatProb = {k: probs[i] for i, (k, v) in enumerate(sorted_sigmas.items())}
        print scatProb.keys()

        this_prob = bisect.bisect_left(scatProb.values(), random.random())
        print 'this prob', this_prob
        self.type = scatProb.keys()[scatProb.values().index(this_prob)]

        if (self.type == 'Moller'):
            # Moller tables are [Ei, Wi, Int(Ei, Wi)]
            self.tables_EW = tables_EW_M
        elif (self.type == 'Gryzinski'):
            # Gryz tables are [ni, Ei, Wi, Int(Ei, Wi)]
            self.tables_EW = tables_EW_G


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

        elif(self.type == 'Bethe'):
            if (self.pathl == 0.):
                print "I'm not telling you how to live your life, but it helps to calculate path lengths before energy losses for CSDA"
            else:
                self.E_loss = self.path * u2n(bethe_mod_sp(self.e.energy, atnd, self.material.get_ns(), \
                                    self.material.get_Es(), self.material.get_Zval(), self.material.get_Eval(), c_pi_efour))



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

        else:
            print 'I did not understand the type of scattering in scatter.calculate_Eloss'


    def compute_sAngles(self):
        if (self.type == 'Rutherford'):
            alpha =  3.4e-3*(self.material.get_Z()**(0.67))/self.e.energy
            self.c2_halfPhi = 1. - alpha*random.random()/(1.+alpha-random.random())
            self.halfTheta = pi*random.random

        elif(self.type == 'Bethe'):

            self.halfTheta = pi*random.random

        elif((self.type == 'Moller') or (e.type == 'Gryzinski')):
            if (self.E_loss == 0.):
                print "I'm not telling you how to live your life, but it helps to calculate the lost energy before the scattering angles for inelastic events"
            else:
                self.c2_halfPhi = 0.5*((1.-(self.Eloss/self.e.energy))**0.5 + 1)
                self.halfTheta = pi*random.random

        else:
            print 'I did not understand the scattering type in scatter.calculate_sAngles'
