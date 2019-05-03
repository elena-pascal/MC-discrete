from math import log, isnan
import numpy as np
import random
import bisect
import sys
import operator
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
from errors import lTooLarge, lTooSmall, E_lossTooSmall, E_lossTooLarge

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
    '''

    def __init__(self, electron, material, free_param, tables_EW_M, tables_EW_G):
        self.e = electron
        self.material = material
        self.free_param = free_param
        self.tables_EW_M = tables_EW_M
        self.tables_EW_G = tables_EW_G

        self.pathl = 0.
        self.type = 0.
        self.E_loss = 0.
        self.c2_halfTheta = 0.
        self.halfPhi = 0.

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
        #self.mfp_total = 1. / self.sigma_total

    def compute_pathl(self):
        '''
        Path length is calculated from the cross section
        path_length = - mean_free_path * log(rn)
        '''
        pathl = UnitScalar(-self.mfp_total * log(random.random()), units = 'angstrom')
        try:
            if (pathl < 1.e-3):
                raise lTooSmall
            elif (pathl > 1.e4):
                raise lTooLarge

        except lTooSmall:
            print ' Fatal error! in compute_pathl in scattering class'
            print ' Value of l less than 0.001 Angstroms.'
            print ' Stopping.'
            sys.exit()
        except lTooLarge:
            print ' Fatal error! in compute_pathl in scattering class'
            print ' Value of l larger than 10000 Angstroms.'
            print ' Stopping.'
            sys.exit()
        self.pathl = pathl

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
        this_prob_pos = bisect.bisect_left(probs, random.random())
        # the type of scattering is the key in the sorted array corresponding to the smallest prob value larger than the random number
        self.type = sorted_sigmas.keys()[this_prob_pos]
        #self.type = 'Rutherford'

        if (self.type == 'Moller'):
            # Moller tables are [Ei, Wi, Int(Ei, Wi)]
            self.tables_EW = self.tables_EW_M
        elif (self.type == 'Gryzinski'):
            # Gryz tables are [ni, Ei, Wi, Int(Ei, Wi)]
            self.tables_EW = self.tables_EW_G


    def compute_Eloss(self):
        '''
        energy loss is calculated from tables for the Moller and Gryz type
        '''
        if (self.type == 'Rutherford'):
            self.E_loss = UnitScalar(0., units='eV')
            #print 'Rutherford elastic scattering has no energy loss'

        elif (self.type == 'Browning'):
            self.E_loss = UnitScalar(0., units='eV')
            #print 'Browning elastic scattering has no energy loss'

        elif(self.type == 'Bethe'):
            if (self.pathl == 0.):
                print "I'm not telling you how to live your life, but it helps to calculate path lengths before energy losses for CSDA"
            else:
                self.E_loss = self.path * u2n(bethe_mod_sp(self.e.energy, atnd, self.material.get_ns(), \
                                    self.material.get_Es(), self.material.get_Zval(), self.material.get_Eval(), c_pi_efour))

        elif(self.type == 'Moller'):
            #wi = np.linear(a, b, 100)
            #ei = np.linear( self.free_param['Ec'], self.e.energy, 1000)
            # integral(E, Wi) is rr * total integral
            # tables_moller are of the form [0, ee, ww[ishell, indx_E, indx_W], Int[[ishell, indx_E, indx_W]]]]
            energies = self.tables_EW_M[1]
            #print 'energies', energies
            Eidx_table = bisect.bisect_left(energies, self.e.energy)
            enlosses = self.tables_EW_M[2][0, Eidx_table, :]
            int_enlosses_table = self.tables_EW_M[3][0, Eidx_table, :]
            integral = random.random() * int_enlosses_table[-1]
            #print 'integrals energy losses', int_enlosses
            #print 'integral', integral
            Wi_table = bisect.bisect_left(int_enlosses_table, integral)
            #print 'Wi_table', Wi_table
            E_loss = enlosses[Wi_table]
            #print 'inside', energylosses[Wi_table]

            try:
                if (E_loss < 1.e-3):
                    raise E_lossTooSmall
                elif (E_loss > self.e.energy*0.5):
                    raise E_lossTooLarge
                else:
                    self.E_loss = E_loss

            except E_lossTooSmall:
                print ' Fatal error! in compute_Eloss for Moller scattering in scattering class'
                print ' Value of energy loss less than 0.001 eV.'
                print ' Stopping.'
                sys.exit()
            except E_lossTooLarge:
                print ' Fatal error! in compute_Eloss for Moller scattering in scattering class'
                print ' Value of energy loss larger than half the electron energy.'
                print ' Stopping.'
                sys.exit()


        elif('Gryzinski' in self.type):
            # the shell is the lefover string after substracting Gryzinski
            shell = self.type.replace('Gryzinski', '')
            ishell = self.material.get_name_s().index(shell)
            # bisect the tables for a random fraction of the maximum
            # energy loss integral for the current energy
            energies = self.tables_EW_G[1]
            #print 'energies',  energies
            Eidx_table = bisect.bisect_left(energies, float(self.e.energy))
            #print 'e losses', self.tables_EW_G[ishell][1]
            #print 'Eidx_table', Eidx_table, float(self.e.energy)
            #print self.tables_EW_G[ishell][2]
            #print self.tables_EW_G[ishell][2][Ei_table, -1]
            enlosses = self.tables_EW_G[2][ishell, Eidx_table, :]
            #print 'enlosses', enlosses

            int_enlosses_table = self.tables_EW_G[3][ishell,Eidx_table, :]
            integral = random.random() * int_enlosses_table[-1]
            #print 'integrals energy losses', self.tables_EW_G[3][ishell, :, :]
            #print 'integral', integral
            Wi_table = bisect.bisect_left(int_enlosses_table, integral)
            #print 'Wi_table', Wi_table, int_enlosses_table, integral
            E_loss = enlosses[Wi_table]
            #print 'E_loss', E_loss

            try:
                if (E_loss < 1.e-3):
                    raise E_lossTooSmall
                elif (E_loss > self.e.energy*0.5):
                    raise E_lossTooLarge
                else:
                    self.E_loss = E_loss

            except E_lossTooSmall:
                print ' Fatal error! in compute_Eloss for Gryzinski scattering in scattering class'
                print ' Value of energy loss less than 0.001 eV.'
                print ' Stopping.'
                sys.exit()
            except E_lossTooLarge:
                print ' Fatal error! in compute_Eloss for Gryzinski scattering in scattering class'
                print ' Value of energy loss larger than half the current energy.'
                print ' Stopping.'
                sys.exit()



        elif(self.type == 'Quinn'):
            self.E_loss = self.material.get_pl_e()

        else:
            print 'I did not understand the type of scattering in scatter.calculate_Eloss'


    def compute_sAngles(self):
        if (self.type == 'Rutherford'):
            alpha =  3.4e-3*(self.material.get_Z()**(0.67))/(float(self.e.energy)*1e-3)
            self.c2_halfTheta = 1. - alpha*random.random()/(1. + alpha-random.random())
            #print self.c2_halfTheta
            self.halfPhi = pi*random.random()

        elif(self.type == 'Bethe'):
            self.halfPhi = pi*random.random()

        elif((self.type == 'Moller') or ('Gryzinski' in self.type)):
            if (self.E_loss == 0.):
                print "you're getting zero energy losses for Moller or Gryz. I suggest you increase the size of the integration table"

            if (float(self.E_loss) < float(self.e.energy)):
                self.c2_halfTheta = 0.5*( (1. - ( float(self.E_loss) / float(self.e.energy) ) )**0.5 + 1.)
                #print self.c2_halfTheta
                self.halfPhi = pi*random.random() # radians
            else:
                self.c2_halfTheta = 0.0
                self.halfPhi = pi*random.random() # radians

        elif(self.type == 'Quinn'):
            self.halfPhi = 0. # we assume plasmon scattering does not affect travelling direction
            self.c2_halfTheta = 0.

        else:
            print 'I did not understand the scattering type in scatter.calculate_sAngles'
