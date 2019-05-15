from math import log, isnan
import numpy as np
import random
import bisect
import sys
import operator

import time

from math import acos

from collections import OrderedDict # sweet sweet ordered dictionaries
from scipy.constants import pi, Avogadro, hbar, m_e, e, epsilon_0
from scipy.interpolate import Rbf

from scimath.units.api import UnitScalar, UnitArray, convert, has_units
from scimath.units.energy import J, eV, KeV
from scimath.units.electromagnetism import coulomb, farad
from scimath.units.length import m, cm, km, angstrom
from scimath.units.time import s
from scimath.units.mass import g, kg
from scimath.units.density import g_per_cm3, kg_per_m3
from scimath.units.substance import mol
from scimath.units.dimensionless import dim

from parameters import u_hbar, u_me, u_e, u_eps0, c_pi_efour, u_bohrr
from electron import electron
from crossSections import ruther_sigma, moller_sigma, gryz_sigma, quinn_sigma
from errors import lTooLarge, lTooSmall, E_lossTooSmall, E_lossTooLarge, wrongUpdateOrder

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
    mfp = 1./(n*sigma) * m**3/cm**2/angstrom
    return mfp

#####################################################################
####################### Scatter class ###############################
#####################################################################
class scatter:
    ''' Scattering can be Rutherford, Browning, Moller, Gryzinski, Quinn or Bethe
    Scattering is defined by
            - incident particle energy
            - material properties
            - the minimum energy for Moller scattering which I call a free parameter
            - tables values
            - scattering parameters to be update by the functions in the class
    '''

    def __init__(self, electron, material, free_param, tables_EW_M, tables_EW_G):
        # incident particle params
        self.i_energy = electron.energy  # incident particle energy

        # material params
        self.m_Z = material.Z()          # atomic number
        self.m_names = material.name_s() # names of the inner shells
        self.m_ns = material.ns()        # number of electrons per inner shell
        self.m_Es = material.Es()        # inner shells energies
        self.m_nval = material.nval()    # number of valence shell electrons
        self.m_Eval = material.Eval()    # valence shell energy
        self.m_atnd = material.atnd()    # atomic number density
        self.m_pl_e = material.pl_e()    # plasmon energy
        self.m_f_e = material.fermi_e()  # Fermi energy

        self.free_param = free_param     # the minimun energy for Moller scattering

        self.tables_EW_M = tables_EW_M
        self.tables_EW_G = tables_EW_G

        # scattering params
        self.pathl = 0.
        self.type = 0.
        self.E_loss = 0.
        self.c2_halfTheta = 0.
        self.halfPhi = 0.

        # some very useful parameters


        # intitalise scattering probabilities dictionary
        self.sigma = {} # dictionary keeping all sigmas
        self.mfp = {} # dictionary keeping all mfp

        ## TODO: decide on sigma or mfp
        self.sigma['Rutherford'] = ruther_sigma(self.i_energy, self.m_Z)
        self.mfp['Rutherford'] = mfp_from_sigma(self.sigma['Rutherford'], self.m_atnd)

        if (self.i_energy > self.m_Eval):
            self.sigma['Moller'] = moller_sigma(self.i_energy, self.free_param, self.m_nval, c_pi_efour)
            self.mfp['Moller'] = mfp_from_sigma(self.sigma['Moller'], self.m_atnd)
            # else the probability of Moller scattering is zero

        for i in range(len(self.m_Es)):
            if (self.i_energy > self.m_Es[i]):
                self.sigma['Gryzinski' + self.m_names[i]] = gryz_sigma(self.i_energy, self.m_Es[i], self.m_ns[i], c_pi_efour)
                self.mfp['Gryzinski' + self.m_names[i]] = mfp_from_sigma(self.sigma['Gryzinski' + self.m_names[i]], self.m_atnd)

        if (self.i_energy > self.m_pl_e):
            self.sigma['Quinn'] = quinn_sigma(self.i_energy, self.m_pl_e, self.m_f_e, self.m_atnd, u_bohrr)
            self.mfp['Quinn'] = mfp_from_sigma(self.sigma['Quinn'], self.m_atnd)

        self.sigma_total = sum(self.sigma.values())
        self.mfp_total = 1. /sum(1./np.array(self.mfp.values()))
        #self.mfp_total = 1. / self.sigma_total

    def compute_pathl(self):
        '''
        Path length is calculated from the cross section
        path_length = - mean_free_path * log(rn)
        '''
        pathl = UnitScalar(-self.mfp_total * log(random.random()), units = 'angstrom')

        try: # ask for forgiveness
            self.pathl = pathl
            # if pathl is too small or too large
            #if (float(pathl) < 1.e-5):
            #    raise lTooSmall
            if (float(pathl) > 1.e4):
                raise lTooLarge

        # except lTooSmall:
        #     print ' Fatal error! in compute_pathl in scattering class'
        #     print ' Value of l is', pathl  ,'less than 0.0001 Angstroms.'
        #     print ' Mean free paths were:', self.mfp
        #     print ' Stopping.'
        #     sys.exit()
        except lTooLarge:
            print ' Fatal error! in compute_pathl in scattering class'
            print ' Value of l is', pathl, 'larger than 1000 Angstroms.'
            print ' Mean free paths were:', self.mfp
            print ' Stopping.'
            sys.exit()



    def det_type(self):
        '''
        bisect for a random number a sorted instance of a dictionary containing
        scattering cumulative probabilities. if R <= cum.prob.scatter.type - > scatter is of type type
        Set the type of scattering after determining type
        '''

        # ordered dictionary of sigmas in reverse order
        sorted_sigmas = OrderedDict(sorted(self.sigma.items(), key=operator.itemgetter(1), reverse=True))

        # probabilities to compare the random number against are cumulative sums of this dictionary
        probs = np.cumsum(sorted_sigmas.values())/np.array(self.sigma_total)

        # bisect the cumulative distribution array with a random number to find the position that random number fits in the array
        #print 'probabilities', sorted_sigmas
        this_prob_pos = bisect.bisect_left(probs, random.random())

        # the type of scattering is the key in the sorted array corresponding to the smallest prob value larger than the random number
        self.type = sorted_sigmas.keys()[this_prob_pos]

        # Moller becomes more unprobable with increase value of Wc


    def compute_Eloss(self):
        '''
        energy loss is calculated from tables for the Moller and Gryz type
        '''
        if (self.type == 'Rutherford'):
            # Rutherford elastic scattering has no energy loss
            self.E_loss = UnitScalar(0., units='eV')
            print 'No energy loss for Rutherford scattering'

        elif (self.type == 'Browning'):
            # Browning elastic scattering has no energy loss
            self.E_loss = UnitScalar(0., units='eV')
            print 'No energy loss for Browning scattering'

        elif(self.type == 'Bethe'):
            if (self.pathl == 0.):
                print "I'm not telling you how to live your life, but it helps to calculate path lengths before energy losses for CSDA"
            else:
                self.E_loss = self.path * u2n(bethe_mod_sp(self.i_energy, self.m_atnd, self.m_ns, \
                                    self.m_Es, self.m_Zval, self.m_Eval, c_pi_efour))

        elif(self.type == 'Moller'):
            # integral(E, Wi) is rr * total integral
            # tables_moller are of the form [0, ee, ww[ishell, indx_E, indx_W], Int[[ishell, indx_E, indx_W]]]]
            # energies = self.tables_EW_M[1]
            Eidx_table = bisect.bisect_left(self.tables_EW_M[1], float(self.i_energy))  # less then value

            int_enlosses_table = self.tables_EW_M[3][0, Eidx_table, :]
            integral = random.random() * int_enlosses_table[-1]
            Wi_table = bisect.bisect_left(int_enlosses_table, integral) 
            # enlosses = self.tables_EW_M[2][0, Eidx_table, :]
            # E_loss = enlosses[Wi_table]
            E_loss = self.tables_EW_M[2][0, Eidx_table, :][Wi_table]

            try:
                self.E_loss = E_loss

                if (E_loss < 1.e-3):
                    raise E_lossTooSmall
                #elif (E_loss > (0.5 * self.i_energy + 100)):
                elif (E_loss >= self.i_energy ):
                    raise E_lossTooLarge

            except E_lossTooSmall:
                print ' Fatal error! in compute_Eloss for Moller scattering in scattering class'
                print ' Value of energy loss less than 0.001 eV.'
                print ' Stopping.'
                sys.exit()
            except E_lossTooLarge:
                print ' Fatal error! in compute_Eloss for Moller scattering in scattering class'
                print ' Value of energy loss larger than half the electron energy.'
                print ' The current energy is:',  self.i_energy
                print ' The corresponding energy in the tables is:',  self.tables_EW_M[1][Eidx_table]
                print ' The current energy lost is:',  E_loss
                print ' The array of energy losses in the tables is:',  self.tables_EW_M[2][0, Eidx_table, :]
                print ' Stopping.'
                sys.exit()



        elif('Gryzinski' in self.type):
            # the shell is the lefover string after substracting Gryzinski
            shell = self.type.replace('Gryzinski', '')
            ishell = self.m_names.index(shell)

            # bisect the tables for a random fraction of the maximum
            # energy loss integral for the current energy

            # energies = self.tables_EW_G[1]
            Eidx_table = bisect.bisect_left(self.tables_EW_G[1], float(self.i_energy))  # less than value
            int_enlosses_table = self.tables_EW_G[3][ishell,Eidx_table, :]
            integral = random.random() * int_enlosses_table[-1]
            Wi_table = bisect.bisect_left(int_enlosses_table, integral)
            # if (Wi_table >= len(enlosses)):
            #     print 'enlosses shape is:', enlosses.shape

            # enlosses = self.tables_EW_G[2][ishell, Eidx_table, :]
            # E_loss = enlosses[Wi_table]
            E_loss = self.tables_EW_G[2][ishell, Eidx_table, :][Wi_table]

            try:
                self.E_loss = E_loss

                if (E_loss < 1.e-3):
                    raise E_lossTooSmall
                #elif (E_loss > ((self.i_energy + max(self.m_Es)*0.5))):
                elif (E_loss >= self.i_energy ):
                    raise E_lossTooLarge

            except E_lossTooSmall:
                print ' Fatal error! in compute_Eloss for Gryzinski scattering in scattering class'
                print ' Value of energy loss less than 0.001 eV.'
                print ' Stopping.'
                sys.exit()
            except E_lossTooLarge:
                print ' Fatal error! in compute_Eloss for Gryzinski scattering in scattering class'
                print ' Value of energy loss larger than half the current energy.'
                print ' The current energy lost is:',  E_loss
                print ' The current energy is:',  self.i_energy
                print ' The corresponding energy in the tables is:',  self.tables_EW_G[1][Eidx_table]
                print ' The array of energy losses in the tables is:',  self.tables_EW_G[2][ishell, Eidx_table, :]
                print ' Try increasing the number of energy bins in the table'
                print ' Stopping.'
                sys.exit()


        elif(self.type == 'Quinn'):
            self.E_loss = self.m_pl_e

        else:
            print 'I did not understand the type of scattering in scatter.calculate_Eloss'


    def compute_sAngles(self):
        if (self.type == 'Rutherford'):
            alpha =  3.4*(self.m_Z**(2./3.))/(float(self.i_energy))
            r = random.random()
            self.c2_halfTheta = 1. - (alpha*r/(1. + alpha - r))
            self.halfPhi = pi*random.random()
            print 'Theta R is', np.degrees(2.*acos(self.c2_halfTheta**0.5))

        elif((self.type == 'Moller') or ('Gryzinski' in self.type)):
            if (self.E_loss == 0.):
                print "you're getting zero energy losses for Moller or Gryz. I suggest you increase the size of the integration table"

            try:
                self.c2_halfTheta = 0.5*( (1. - (( self.E_loss / float(self.i_energy) ) )**0.5) + 1.)
                print 'E_loss is ', self.E_loss, self.i_energy
                print 'Theta is', np.degrees(2.*acos(self.c2_halfTheta**0.5))


                if (self.E_loss > self.i_energy):
                    raise wrongUpdateOrder
            except wrongUpdateOrder:
                print ' You might be updating energy before calculating the scattering angle'
                print ' Stopping.'
                sys.exit()


            self.halfPhi = pi*random.random() # radians


        #elif(self.type == 'Quinn'):
            # we assume plasmon scattering does not affect travelling direction
            # TODO: small angles for Quinn
        #    self.halfPhi = 0.

        #else:
        #    print 'I did not understand the scattering type in scatter.calculate_sAngles'
