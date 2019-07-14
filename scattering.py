from math import log, isnan, acos
import numpy as np
import random
import bisect
import sys
import operator

from collections import OrderedDict # sweet sweet ordered dictionaries
from scipy.constants import pi, Avogadro, hbar, m_e, e, epsilon_0

from scimath.units.api import has_units
from scimath.units.length import angstrom, cm,  m

from electron import electron
from crossSections import ruther_sigma, moller_sigma, gryz_sigma, quinn_sigma
from stoppingPowers import bethe_cl_sp, bethe_mod_sp, bethe_mod_sp_k
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
####################### Discrete inelastic scatter class ############
#####################################################################
class scatter_discrete:
    ''' Scattering can be Rutherford, Browning, Moller, Gryzinski, Quinn
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
        self.m_Z = material.params['Z']          # atomic number
        self.m_names = material.params['name_s'] # names of the inner shells
        self.m_ns = material.params['ns']        # number of electrons per inner shell
        self.m_Es = material.params['Es']        # inner shells energies
        self.m_nval = material.params['n_val']   # number of valence shell electrons
        self.m_Eval = material.params['E_val']   # valence shell energy
        self.m_atnd = material.atnd              # atomic number density
        self.m_pl_e = material.plasmon_e         # plasmon energy
        self.m_f_e = material.fermi_e            # Fermi energy

        self.free_param = free_param     # the minimun energy for Moller scattering

        self.tables_EW_M = tables_EW_M
        self.tables_EW_G = tables_EW_G

        # scattering params
        self.pathl = 0.
        self.type = 0.
        self.E_loss = 0.
        self.c2_halfTheta = 0.
        self.halfPhi = 0.

        # intitalise scattering probabilities dictionary
        self.sigma = {} # dictionary keeping all sigmas
        #self.mfp = {} # dictionary keeping all mfp

        ## TODO: decide on sigma or mfp. Is sigma the inverse mean free path?
        self.sigma['Rutherford'] = ruther_sigma(self.i_energy, self.m_Z)
        #self.mfp['Rutherford'] = mfp_from_sigma(self.sigma['Rutherford'], self.m_atnd)

        # if the energy is larger than the valence energy consider Moller scattering
        if (self.i_energy >= self.m_Eval):
            self.sigma['Moller'] = moller_sigma(self.i_energy, self.free_param, self.m_nval)
            #self.mfp['Moller'] = mfp_from_sigma(self.sigma['Moller'], self.m_atnd)
            # else the probability of Moller scattering is the default zero

        for i in range(len(self.m_Es)):
            if (self.i_energy > self.m_Es[i]):
                self.sigma['Gryzinski' + self.m_names[i]] = gryz_sigma(self.i_energy, self.m_Es[i], self.m_ns[i])
                #self.mfp['Gryzinski' + self.m_names[i]] = mfp_from_sigma(self.sigma['Gryzinski' + self.m_names[i]], self.m_atnd)

        if (self.i_energy > self.m_pl_e):
            self.sigma['Quinn'] = quinn_sigma(self.i_energy, self.m_pl_e, self.m_f_e, self.m_atnd)
            #self.mfp['Quinn'] = mfp_from_sigma(self.sigma['Quinn'], self.m_atnd)

        self.sigma_total = sum(self.sigma.values())
        #self.mfp_total = 1. /sum(1./np.array(self.mfp.values()))
        self.mfp_total = mfp_from_sigma( self.sigma_total, self.m_atnd)


    def compute_pathl(self):
        '''
        Path length is calculated from the cross section
        path_length = - mean_free_path * log(rn)
        '''
        pathl = -self.mfp_total * log(random.random())

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
        except lTooLarge as err:
            print ('-----------------------------------------------------')
            print (' Fatal error:', err)
            print (' in compute_pathl in scattering class')
            print (' Value of l is', pathl, 'larger than 10000 Angstroms.')
            print (' Mean free paths were:', mfp_from_sigma(self.sigma_total, self.m_atnd))
            print (' Stopping.')
            sys.exit()



    def det_type(self):
        '''
        bisect for a random number a sorted instance of a dictionary containing
        scattering cumulative probabilities. if R <= cum.prob.scatter.type - > scatter is of type type
        Set the type of scattering after determining type
        '''

        # ordered dictionary of sigmas in reverse order
        sorted_sigmas = OrderedDict(sorted(self.sigma.items(), key=operator.itemgetter(1), reverse=True))

        #extract = lambda x, y: dict(zip(x, map(y.get, x)))

        # probabilities to compare the random number against are cumulative sums of this dictionary
        #probs = np.cumsum(list(sorted_sigmas.values()))/self.sigma_total
        probs = np.cumsum(list(sorted_sigmas.values()))/sum(sorted_sigmas.values())
        # bisect the cumulative distribution array with a random number to find the position that random number fits in the array
        this_prob_pos = bisect.bisect_left(probs, random.random())

        # the type of scattering is the key in the sorted array corresponding to the smallest prob value larger than the random number
        self.type = list(sorted_sigmas.keys())[this_prob_pos]
        #self.type = 'Gryzinski1s'

        # Moller becomes more unprobable with increase value of Wc


    def compute_Eloss(self):
        '''
        energy loss is calculated from tables for the Moller and Gryz type
        '''

        if(self.type == 'Moller'):
            # integral(E, Wi) is rr * total integral
            # tables_moller are of the form [0, ee, ww[ishell, indx_E, indx_W], Int[[ishell, indx_E, indx_W]]]]
            # energies = self.tables_EW_M[1]
            Eidx_table = bisect.bisect_left(self.tables_EW_M[0], self.i_energy) - 1  # less than value
            #print ('e index', Eidx_table)
            #print ('int tables at line E',  self.tables_EW_M[0][2])
            int_W_table = self.tables_EW_M[1][2][Eidx_table]
            integral = random.random() * int_W_table[-1]
            Widx_table = bisect.bisect_left(int_W_table, integral)
            E_loss = self.tables_EW_M[1][1][Eidx_table][Widx_table]

            try:
                self.E_loss = E_loss
                # if (E_loss < 1.e-3):
                #     raise E_lossTooSmall
                if (E_loss >= self.i_energy ):
                    raise E_lossTooLarge

            # except E_lossTooSmall:
            #     print ' Fatal error! in compute_Eloss for Moller scattering in scattering class'
            #     print ' Value of energy loss less than 0.001 eV.'
            #     print ' Stopping.'
            #     sys.exit()
            except E_lossTooLarge as err:
                print (' Fatal error:', err)
                print (' in compute_Eloss for Moller scattering in scattering class')
                print (' Value of energy loss larger than half the electron energy.')
                print (' The current energy is:',  self.i_energy)
                print (' The corresponding energy in the tables is:',  self.tables_EW_M[0][Eidx_table])
                print (' The current energy lost is:',  E_loss)
                print (' The array of energy losses in the tables is:',  self.tables_EW_M[1][1][Eidx_table][Widx_table-1:Widx_table+1])
                print (' Stopping.')
                sys.exit()



        elif('Gryzinski' in self.type):
            # the shell is the lefover string after substracting Gryzinski
            shell = self.type.replace('Gryzinski', '')
            ishell = self.m_names.index(shell) + 1 # in tables index 0 is the energy list

            # bisect the tables for a random fraction of the maximum
            # energy loss integral for the current energy

            # energies = self.tables_EW_G[1]
            Eidx_table = bisect.bisect_left(self.tables_EW_G[0], self.i_energy) - 1  # less than value
            #print ('energy is', self.i_energy, 'from table:', self.tables_EW_G[0], self.tables_EW_G[0][Eidx_table])
            int_W_table = self.tables_EW_G[ishell][2][Eidx_table]
            #print ('integral tables', int_W_table)
            integral = random.random() * int_W_table[-1]
            Widx_table = bisect.bisect_left(int_W_table, integral)
            E_loss = self.tables_EW_G[ishell][1][Eidx_table][Widx_table]

            try:
                self.E_loss = E_loss
                # if (E_loss < 1.e-3):
                #     raise E_lossTooSmall
                #elif (E_loss > ((self.i_energy + max(self.m_Es)*0.5))):
                if (E_loss >= self.i_energy ):
                    raise E_lossTooLarge

            # except E_lossTooSmall:
            #     print ' ---------------------------------------------------------------------------'
            #     print ' Fatal error! in compute_Eloss for Gryzinski scattering in scattering class'
            #     print ' Value of energy loss less than 0.001 eV.'
            #     print ' Stopping.'
            #     sys.exit()
            except E_lossTooLarge as err:
                print (' --------------------------------------------------------------------------')
                print (' Fatal error:', err)
                print (' in compute_Eloss for Gryzinski scattering in scattering class')
                print (' Value of energy loss larger than half the current energy.')
                print (' The current energy is:',  self.i_energy)
                print (' The corresponding energy in the tables is:',  self.tables_EW_G[0][Eidx_table])
                print (' The current energy lost is:',  E_loss)
                print (' The neigbour energy losses in the tables are:',  self.tables_EW_G[ishell][1][Eidx_table][Widx_table-1:Widx_table+1])
                print (' Try increasing the number of energy bins in the table')
                print (' Stopping.')
                sys.exit()


        elif(self.type == 'Quinn'):
            self.E_loss = self.m_pl_e

        else:
            print (' I did not understand the type of scattering in scatter.calculate_Eloss')

    def compute_sAngles(self):
        if (self.type == 'Rutherford'):
            alpha =  3.4*(self.m_Z**(2./3.))/(float(self.i_energy))
            r = random.random()
            self.c2_halfTheta = 1. - (alpha*r/(1. + alpha - r))
            self.halfPhi = pi*random.random()
            #print 'Theta R is', np.degrees(2.*acos(self.c2_halfTheta**0.5))

        elif((self.type == 'Moller') or ('Gryzinski' in self.type)):
            if (self.E_loss == 0.):
                print (" you're getting zero energy losses for Moller or Gryz. I suggest you increase the size of the integration table")

            try:
                self.c2_halfTheta = 0.5*( (1. - (( self.E_loss / float(self.i_energy) ) )**0.5) + 1.)
                if (self.E_loss > self.i_energy):
                    raise wrongUpdateOrder
            except wrongUpdateOrder as err:
                print ( 'Error:', err)
                print (' You might be updating energy before calculating the scattering angle')
                print (' Stopping.')
                sys.exit()


            self.halfPhi = pi*random.random() # radians


        #elif(self.type == 'Quinn'):
            # we assume plasmon scattering does not affect travelling direction
            # TODO: small angles for Quinn
        #    self.halfPhi = 0.

        #else:
        #    print 'I did not understand the scattering type in scatter.calculate_sAngles'




#######################  with units  #########################################
from parameters import u_bohr_r, u_pi_efour

class scatter_discrete_wUnits(scatter_discrete):
    ''' scatter_discerte_wUnits inherits the class scatter_discrete
        It is different because we try to track units here.
        The way to do that is to use inputs with units
        and give the physical parameters with units

        Practically the only difference is that we call the sigma function with
        explicit unitted parameters
    '''

    def __init__(self, electron, material, free_param, tables_EW_M, tables_EW_G):
        # incident particle params
        self.i_energy = electron.energy  # incident particle energy

        # material params
        self.m_Z = material.params['Z']          # atomic number
        self.m_names = material.params['name_s'] # names of the inner shells
        #self.m_ns = material.params['ns']        # number of electrons per inner shell
        self.m_Es = material.params['Es']        # inner shells energies
        self.m_nval = material.params['n_val']    # number of valence shell electrons
        self.m_Eval = material.params['E_val']    # valence shell energy
        self.m_atnd = material.atnd    # atomic number density
        self.m_pl_e = material.plasmon_e    # plasmon energy
        self.m_f_e = material.fermi_e  # Fermi energy

        self.free_param = free_param     # the minimun energy for Moller scattering

        self.tables_EW_M = tables_EW_M
        self.tables_EW_G = tables_EW_G

        # scattering params
        self.pathl = 0.
        self.type = 0.
        self.E_loss = 0.
        self.c2_halfTheta = 0.
        self.halfPhi = 0.

        # intitalise scattering probabilities dictionary
        self.sigma = {} # dictionary keeping all sigmas

        self.sigma['Rutherford'] = ruther_sigma(self.i_energy, self.m_Z)

        # if the energy is larger than the valence energy consider Moller scattering
        if (self.i_energy >= self.m_Eval):
            self.sigma['Moller'] = moller_sigma(self.i_energy, self.free_param, self.m_nval, u_pi_efour)
            #self.mfp['Moller'] = mfp_from_sigma(self.sigma['Moller'], self.m_atnd)
            # else the probability of Moller scattering is zero

        for i in range(len(self.m_Es)):
            if (self.i_energy > self.m_Es[i]):
                self.sigma['Gryzinski' + self.m_names[i]] = gryz_sigma(self.i_energy, self.m_Es[i], self.m_ns[i], u_pi_efour)
                #self.mfp['Gryzinski' + self.m_names[i]] = mfp_from_sigma(self.sigma['Gryzinski' + self.m_names[i]], self.m_atnd)

        if (self.i_energy > self.m_pl_e):
            self.sigma['Quinn'] = quinn_sigma(self.i_energy, self.m_pl_e, self.m_f_e, self.m_atnd, u_bohr_r)
            #self.mfp['Quinn'] = mfp_from_sigma(self.sigma['Quinn'], self.m_atnd)

        self.sigma_total = sum(self.sigma.values())
        #self.mfp_total = 1. /sum(1./np.array(self.mfp.values()))
        self.mfp_total = mfp_from_sigma( self.sigma_total, self.m_atnd)

        self.halfPhi = pi*random.random() # radians




################################################################################
####################### Continuous inelastic scatter class #####################
#######################           3 Bethe models           #####################
################################################################################

#### 1) classical Bethe
class scatter_continuous_classical:
    ''' This is the CSDA scattering mode
    Rutherford is the elastic scattering and accounts for angular deviation
    and classical Bethe is the continuous energy loss
    '''

    def __init__(self, electron, material):
        # incident particle params
        self.i_energy = electron.energy  # incident particle energy

        # material params
        self.m_Z = material.params['Z']  # atomic number
        self.m_atnd = material.atnd      # atomic number density

        # scattering params
        self.pathl = 0.
        self.E_loss = 0.
        self.c2_halfTheta = 0.
        self.halfPhi = 0.


        # intitalise scattering probabilities dictionary
        self.sigma = {} # dictionary keeping all sigmas
        self.mfp = {}   # dictionary keeping all mfp

        ## TODO: decide on sigma or mfp
        self.sigma['Rutherford'] = ruther_sigma(self.i_energy, self.m_Z)
        self.mfp['Rutherford'] = mfp_from_sigma(self.sigma['Rutherford'], self.m_atnd)


    def compute_pathl(self):
        '''
        Path length is calculated from the cross section
        path_length = - mean_free_path * log(rn)
        '''
        pathl = -self.mfp['Rutherford'] * log(random.random())

        try: # ask for forgiveness
            self.pathl = pathl
            if (float(pathl) > 1.e4):
                raise lTooLarge

        except lTooLarge as err:
            print (' Error:', err)
            print (' Fatal error! in compute_pathl in scattering class')
            print (' Value of l is', pathl, 'larger than 1000 Angstroms.')
            print (' Mean free paths were:', mfp_from_sigma(self.sigma, self.m_atnd))
            print (' Stopping.')
            sys.exit()



    def compute_Eloss(self):
        '''
        energy loss is calculated from Bethe's CSDA
        '''

        if (self.pathl == 0.):
            print (" I'm not telling you how to live your life, but it helps to calculate path lengths before energy losses for CSDA")
        else:
            E_loss = self.pathl * bethe_cl_sp(self.m_Z, self.i_energy, self.m_atnd)

            try:
                self.E_loss = E_loss
                # if (E_loss < 1.e-5):
                #     raise E_lossTooSmall
                if (E_loss >= self.i_energy ):
                    raise E_lossTooLarge

            # TODO: set lower limit of pathl ?
            # except E_lossTooSmall:
            #     print ' ---------------------------------------------------------------------------'
            #     print ' Fatal error! in compute_Eloss for Bethe scattering in scattering class'
            #     print ' Value of energy loss less than 0.001 eV.'
            #     print ' Stopping.'
            #     sys.exit()
            except E_lossTooLarge as err:
                print (' --------------------------------------------------------------------------')
                print (' Fatal error:', err)
                prinnt(' in compute_Eloss for Bethe scattering in scattering class')
                print (' Value of energy loss larger than half the current energy.')
                print (' The current energy lost is:',  E_loss)
                print (' The current energy is:',  self.i_energy)
                print (' Stopping.')
                sys.exit()

    def compute_sAngles(self):
        alpha =  3.4*(self.m_Z**(2./3.))/(float(self.i_energy))
        r = random.random()
        self.c2_halfTheta = 1. - (alpha*r/(1. + alpha - r))
        self.halfPhi = pi*random.random()


# Joy and Luo Bethe as extended from the classical one
class scatter_continuous_JL(scatter_continuous_classical):
    ''' This is the CSLA scattering mode
    Rutherford is the elastic scattering and accounts for angular deviation
    and Joy and Luo modiefied form of Bethe is the continuous energy loss
    '''

    def __init__(self, electron, material):
        # incident particle params
        self.i_energy = electron.energy  # incident particle energy

        # material params
        self.m_Z = material.params['Z']           # atomic number
        self.m_k = material.params['bethe_k']     # k value for Joy and Luo equation
        self.m_atnd = material.atnd    # atomic number density


        # scattering params
        self.pathl = 0.
        self.type = 0.
        self.E_loss = 0.
        self.c2_halfTheta = 0.
        self.halfPhi = 0.


        # intitalise scattering probabilities dictionary
        self.sigma = {} # dictionary keeping all sigmas
        self.mfp = {} # dictionary keeping all mfp

        ## TODO: decide on sigma or mfp
        self.sigma['Rutherford'] = ruther_sigma(self.i_energy, self.m_Z)
        self.mfp['Rutherford'] = mfp_from_sigma(self.sigma['Rutherford'], self.m_atnd)

    def compute_Eloss(self):
        '''
        energy loss is calculated from Bethe's CSDA
        '''

        if (self.pathl == 0.):
            print (" I'm not telling you how to live your life, but it helps to calculate path lengths before energy losses for CSDA")
        else:
            E_loss = self.pathl * bethe_mod_sp_k(self.m_Z, self.i_energy, self.m_atnd, self.m_k)
            try:
                self.E_loss = E_loss
                # if (E_loss < 1.e-6):
                #     raise E_lossTooSmall
                #elif (E_loss > ((self.i_energy + max(self.m_Es)*0.5))):
                if (E_loss >= self.i_energy ):
                    raise E_lossTooLarge
            #
            # except E_lossTooSmall:
            #     print ' ---------------------------------------------------------------------------'
            #     print ' Fatal error! in compute_Eloss for Bethe scattering in scattering class'
            #     print ' Value of energy loss less than 0.001 eV.'
            #     print ' Path lenght was', self.pathl
            #     print ' Stopping.'
            #     sys.exit()
            except E_lossTooLarge as err:
                print (' --------------------------------------------------------------------------')
                print (' Fatal error:', err)
                print (' in compute_Eloss for Bethe scattering in scattering class')
                print (' Value of energy loss larger than half the current energy.')
                print (' The current energy lost is:',  E_loss)
                print (' The current energy is:',  self.i_energy)
                print (' Stopping.')
                sys.exit()


# explicit shells contributions Bethe as extention from the classical one
class scatter_continuous_explicit(scatter_continuous_classical):
    ''' This is the CSLA scattering mode
    Rutherford is the elastic scattering and accounts for angular deviation
    and the explicit modified version of Bethe is the continuous energy loss
    '''

    def __init__(self, electron, material):
        # incident particle params
        self.i_energy = electron.energy  # incident particle energy

        # material params
        self.m_Z = material.params['Z']           # atomic number
        self.m_Es = material.params['Es']         # inner shells energies
        self.m_ns = material.params['ns']         # number of electrons per inner shell
        self.m_nval = material.params['n_val']    # number of valence shell electrons
        self.m_Eval = material.params['E_val']    # valence shell energy
        self.m_atnd = material.atnd    # atomic number density


        # scattering params
        self.pathl = 0.
        self.type = 0.
        self.E_loss = 0.
        self.c2_halfTheta = 0.
        self.halfPhi = 0.


        # intitalise scattering probabilities dictionary
        self.sigma = {} # dictionary keeping all sigmas
        self.mfp = {} # dictionary keeping all mfp

        ## TODO: decide on sigma or mfp
        self.sigma['Rutherford'] = ruther_sigma(self.i_energy, self.m_Z)
        self.mfp['Rutherford'] = mfp_from_sigma(self.sigma['Rutherford'], self.m_atnd)


    def compute_Eloss(self):
        '''
        energy loss is calculated from Bethe's CSDA
        '''

        if (self.pathl == 0.):
            print (" I'm not telling you how to live your life, but it helps to calculate path lengths before energy losses for CSDA")
        else:
            E_loss = self.pathl * bethe_mod_sp(self.i_energy, self.m_atnd, self.m_ns, \
                                    self.m_Es, self.m_nval, self.m_Eval)

            try:
                self.E_loss = E_loss
                # TODO: is there a lower limit
                # if (E_loss < 1.e-5):
                #      raise E_lossTooSmall
                if (E_loss >= self.i_energy ):
                    raise E_lossTooLarge

            # except E_lossTooSmall:
            #     print ' ---------------------------------------------------------------------------'
            #     print ' Fatal error! in compute_Eloss for Bethe scattering in scattering class'
            #     print ' Value of energy loss less than 0.001 eV.'
            #     print ' Path lenght is', self.pathl
            #     print ' Stopping.'
            #     sys.exit()
            except E_lossTooLarge as err:
                print (' --------------------------------------------------------------------------')
                print (' Fatal error;', err)
                print (' in compute_Eloss for Bethe scattering in scattering class')
                print (' Value of energy loss larger than half the current energy.')
                print (' The current energy lost is:',  E_loss)
                print (' The current energy is:',  self.i_energy)
                print (' Stopping.')
                sys.exit()



#######################  with units  #########################################
## 1)
class scatter_continuous_classical_wUnits(scatter_continuous_classical):
    def compute_Eloss(self):
        '''
        energy loss is calculated from Bethe's CSDA
        '''

        if (self.pathl == 0.):
            print (" I'm not telling you how to live your life, but it helps to calculate path lengths before energy losses for CSDA")
        else:
            self.E_loss = self.pathl * bethe_cl_sp(self.m_Z, self.i_energy, self.m_atnd, u_pi_efour)

## 2)
class scatter_continuous_JL_wUnits(scatter_continuous_JL):
    def compute_Eloss(self):
        '''
        energy loss is calculated from Bethe's CSDA
        '''

        if (self.pathl == 0.):
            print ("I'm not telling you how to live your life, but it helps to calculate path lengths before energy losses for CSDA")
        else:
            self.E_loss = self.pathl * bethe_mod_sp_k(self.m_Z, self.i_energy, self.m_atnd, self.m_k, u_pi_efour)

## 3)
class scatter_continuous_explicit_wUnits(scatter_continuous_explicit):
    def compute_Eloss(self):
        '''
        energy loss is calculated from Bethe's CSDA
        '''

        if (self.pathl == 0.):
            print (" I'm not telling you how to live your life, but it helps to calculate path lengths before energy losses for CSDA")
        else:
            self.E_loss = self.pathl * bethe_mod_sp(self.i_energy, self.m_atnd, self.m_ns, \
                                    self.m_Es, self.m_nval, self.m_Eval, u_pi_efour)
