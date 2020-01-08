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

from MC.electron import electron
from MC.crossSections import alpha, ruther_sigma, ruther_N_sigma, ruther_N_sigma_wDefl,  moller_sigma, gryz_sigma, quinn_sigma
from MC.stoppingPowers import bethe_cl_sp, bethe_mod_sp, bethe_mod_sp_k, moller_sp
from MC.errors import lTooLarge, lTooSmall, E_lossTooSmall, E_lossTooLarge, wrongUpdateOrder, ElossGTEnergy

@has_units
def mfp_from_sigma(sigma, n):
    """ Calculate the mean free path from the total cross section

        Parameters
        ----------
        sigma  : array : units = cm**2
                total cross section

        n      : array : units = m**-3
                number density

        Returns
        -------
        mfp    : array : units = angstrom
    """
    mfp = 1./(n*sigma) * m**3/cm**2/angstrom
    return mfp


def pickFromSigmas(sigma_dict):
    '''
    From a dictionary of sigmas pick a scattering type.

    Bisect for a random number a sorted instance of a dictionary containing
    scattering cumulative probabilities. if R <= cum.prob.scatter.type
    - > scatter is of type type
    '''
    # ordered dictionary of sigmas in reverse order
    sorted_sigmas = OrderedDict(sorted(sigma_dict.items(), key=operator.itemgetter(1), reverse=True))
    #sorted_sigmas['Quinn'] = sigma_dict['Quinn']

    #extract = lambda x, y: dict(zip(x, map(y.get, x)))
    #sorted_sigmas = extract(['Rutherford', 'Moller', 'Gryzinski1s'], sorted_sigmas)
    #print ('after', sorted_sigmas)

    # probabilities to compare the random number against are cumulative sums of this dictionary
    # [pR = sigmaR/sigmaT, pQ = (sigmaR+sigmaQ)/sigmaT, ... ]
    probs = np.cumsum(list(sorted_sigmas.values()))/sum(sorted_sigmas.values())

    # bisect the cumulative distribution array with a random number to find
    # the position that random number fits in the array
    pickedProb = bisect.bisect_left(probs, random.random())

    # the type of scattering is the key in the sorted array corresponding to the smallest prob value larger than the random number
    return  list(sorted_sigmas.keys())[pickedProb]



def pickTable(table, energy):
    '''
    From Moller tables containing the integral under the excitation function
    for a list of incident energies and energies losses
    pick an energy loss value

    input:
        table  : a type table object
        energy : float value

    return:
        (E_loss, equivalent energy in table, neighbouring energy losses in table)
    '''

    # read energy dataframe from h5 file
    energies_table = table.Es

    # find the index in the table for this energy
    Eidx_table = bisect.bisect_left(table.Es, energy)   # less than value
    # NOTE: this is not a bad implementation, except for the cases when the energy is
    # exactly equal to the min value and we end up with index -1
    # I avoided that by setting electrons of energies <= starting value to be absorbed

    # get the column in the table for this energy
    CDF_list = table.table[Eidx_table]

    # let's pick a value. integral(E, Wi) is rr * total integral
    Widx_table = bisect.bisect_left(CDF_list, random.random())

    # which is an energy loss of
    E_loss = table.Ws[Widx_table]

    return (E_loss, energies_table[Eidx_table])


def Rutherford_halfPol(energy, Z):
    '''
    compute the polar angle for Rutherford scattering
    using the analytical solution of the integral
    of the differential cross section.

    See eq. 3.10 in Joy's.

    Returns an array of values of the same length as the energy array.

    Parameters
    ----------
    energy : array : units = KeV

    Z      : array : units = dim

    Returns
    -------
    cos^2(halfTheta) : array : units = dim
    '''
    # if energy arrives a scalar
    if np.isscalar(energy):
        rn = random.random()
    else: # energy is a numpy array
        rn = np.array([random.random() for _ in range(energy.size)])

    alphaR = alpha(energy, Z)
    return 1. - (alphaR*rn/(1. + alphaR - rn))



def binaryCollModel(energy, e_loss):
    '''
    Binary collision model is used for detemining the
    scattering polar angle in Moller and Gryzinski type events

    Note: that the binary collision model assumes the scattering angle
    to be [0, pi/2]

    Parameters
    ----------
    energy : array : units = KeV

    e_loss : array : units = KeV

    Returns
    -------
    cos^2(halfTheta) : array : units = dim
    '''
    return 0.5*( (1 - (e_loss / energy))**0.5 + 1)


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

    def __init__(self, electron, material, free_param, tables):
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

        self.table_EW_M = tables['Moller']
        self.tables_EW_G = tables['Gryz']

        # scattering params
        self.pathl = None
        self.type = 'Rutherford' # first entry is elastic
        self.E_loss = None
        self.c2_halfTheta = None
        self.halfPhi = None

        # intitalise scattering probabilities dictionary
        self.sigma = {} # dictionary keeping all sigmas

        self.sigma['Rutherford'] = ruther_N_sigma(self.i_energy, self.m_Z)

        # if the energy is larger than the valence energy consider Moller scattering
        self.sigma['Moller'] = moller_sigma(self.i_energy, self.free_param, self.m_nval)

        for shell in self.m_names:
            self.sigma['Gryzinski' + shell] = gryz_sigma(self.i_energy, self.m_Es[shell], self.m_ns[shell])

        self.sigma['Quinn'] = quinn_sigma(self.i_energy, self.m_pl_e, self.m_f_e, self.m_atnd)

        self.sigma_total = sum(self.sigma.values())

        self.mfp_total = mfp_from_sigma( self.sigma_total, self.m_atnd)

        # output lists object
        self.scat_output = electron.scat_output

    def compute_pathl(self):
        '''
        Path length is calculated from the cross section
        path_length = - mean_free_path * log(rn)
        '''
        pathl = -self.mfp_total * log(random.random())

        # check if the value is not ridiculous
        assert pathl < 1e4, "Mean free path larger than 10000 A: %s > %s" %(pathl, 1e4)

        # assign it
        self.pathl = pathl

        # save path length if we want it
        self.scat_output.addToList('pathl', self.pathl)


    def det_type(self):
        '''
        Set the type of scattering after determining type
        '''
        self.type = pickFromSigmas(self.sigma)
        # NOTE: Moller becomes more unprobable with increase value of Wc

        # save scatter type if we want it
        self.scat_output.addToList('type', self.type)

    def compute_Eloss_and_angles(self):
        '''
        Compute both the energy loss and the scattering angles.

        Energy loss is calculated from tables for the Moller and Gryz type
        '''
        ######## Rutherford ########
        if(self.type == 'Rutherford'):
            self.E_loss = 0.
            self.c2_halfTheta = Rutherford_halfPol(self.i_energy, self.m_Z)

            # save energy if we want it
            self.scat_output.addToList('E', self.i_energy)

            # save energy loss if we want it
            self.scat_output.addToList('E_loss', self.E_loss)

            # save polar angle if we want it
            self.scat_output.addToList('pol_angle', self.c2_halfTheta)

        ##### Moller ###############
        elif(self.type == 'Moller'):
            E_loss, tables_e = pickTable(self.table_EW_M, self.i_energy)

            assert E_loss < self.i_energy, "Energy loss larger than electron energy: %s > %s" %(E_loss, self.i_energy)
            self.E_loss = E_loss

            # polar angle
            self.c2_halfTheta = binaryCollModel(self.i_energy, self.E_loss)

            # save energy if we want it
            self.scat_output.addToList('E', self.i_energy)

            # save energy loss if we want it
            self.scat_output.addToList('E_loss', self.E_loss)

            # save apolar angle if we want it
            self.scat_output.addToList('pol_angle', self.c2_halfTheta)

        ##### Gryzinski ###########
        elif('Gryzinski' in self.type):
            # the shell name is the lefover string after substracting Gryzinski
            shell = self.type.replace('Gryzinski', '')

            E_loss, tables_e = pickTable(self.tables_EW_G[0], self.i_energy)

            assert E_loss < self.i_energy, "Energy loss larger than electron energy: %s > %s" %(E_loss, self.i_energy)
            self.E_loss = E_loss

            # polar angle
            self.c2_halfTheta = binaryCollModel(self.i_energy, self.E_loss)

            # save energy if we want it
            self.scat_output.addToList('E', self.i_energy)

            # save energy loss if we want it
            self.scat_output.addToList('E_loss', self.E_loss)

            # save pol angle if we want it
            self.scat_output.addToList('pol_angle', self.c2_halfTheta)

        ##### Quinn ###########
        elif(self.type == 'Quinn'):
            self.E_loss = self.m_pl_e

            # for plasmon scattering assume no change in direction
            self.c2_halfTheta = 1.

            # save energy if we want it
            self.scat_output.addToList('E', self.i_energy)

            # save energy loss if we want it
            self.scat_output.addToList('E_loss', self.E_loss)

            # save polar angle if we want it
            self.scat_output.addToList('pol_angle', self.c2_halfTheta)

        else:
            print (' I did not understand the type of scattering in scatter.calculate_Eloss')

        # polar angle is the same for all scatterings
        self.halfPhi = pi*random.random()

        # save azimuthal angle if we want it
        self.scat_output.addToList('az_angle', self.halfPhi)



#######################  with units  #########################################
from MC.parameters import u_bohr_r, u_pi_efour

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
        self.m_Z = material.params['Z']           # atomic number

        self.m_names = material.params['name_s']  # names of the inner shells
        self.m_ns = material.params['ns']         # number of electrons per inner shell
        self.m_Es = material.params['Es']         # inner shells energies

        self.m_nval = material.params['n_val']    # number of valence shell electrons
        self.m_Eval = material.params['E_val']    # valence shell energy

        self.m_atnd = material.atnd               # atomic number density
        self.m_pl_e = material.plasmon_e          # plasmon energy
        self.m_f_e = material.fermi_e             # Fermi energy

        self.free_param = free_param     # the minimun energy for Moller scattering

        self.tables_EW_M = tables_EW_M
        self.tables_EW_G = tables_EW_G

        # scattering params
        self.pathl = None
        self.type = None
        self.E_loss = None
        self.c2_halfTheta = None
        self.halfPhi = None

        # intitalise scattering probabilities dictionary
        self.sigma = {} # dictionary keeping all sigmas

        self.sigma['Rutherford'] = ruther_sigma(self.i_energy, self.m_Z)

        # if the energy is larger than the valence energy consider Moller scattering
        self.sigma['Moller'] = moller_sigma(self.i_energy, self.free_param, self.m_nval, u_pi_efour)
        # else the probability of Moller scattering is zero

        for i in range(len(self.m_Es)):
            self.sigma['Gryzinski' + self.m_names[i]] = gryz_sigma(self.i_energy, self.m_Es[i], self.m_ns[i], u_pi_efour)

        if (self.i_energy > self.m_pl_e):
            self.sigma['Quinn'] = quinn_sigma(self.i_energy, self.m_pl_e, self.m_f_e, self.m_atnd, u_bohr_r)

        self.sigma_total = sum(self.sigma.values())

        self.mfp_total = mfp_from_sigma( self.sigma_total, self.m_atnd)

        self.halfPhi = pi*random.random() # radians




################################################################################
####################### Continuous inelastic scatter class #####################
#######################           3 Bethe models           #####################
################################################################################

#### 1) classical Bethe
class scatter_continuous_classical:
    ''' This is the CSDA scattering mode classical model
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
        self.pathl = None
        self.E_loss = None
        self.c2_halfTheta = None
        self.halfPhi = None


        # intitalise scattering probabilities dictionary
        self.sigma = {} # dictionary keeping all sigmas
        self.mfp = {}   # dictionary keeping all mfp

        ## TODO: decide on sigma or mfp
        #self.sigma['Rutherford'] = ruther_sigma(self.i_energy, self.m_Z)
        self.sigma['Rutherford'] = ruther_N_sigma_wDefl(self.i_energy, self.m_Z)
        self.mfp['Rutherford'] = mfp_from_sigma(self.sigma['Rutherford'], self.m_atnd)

        # scattering output object
        self.scat_output = electron.scat_output

    def compute_pathl(self):
        '''
        Path length is calculated from the cross section
        path_length = - mean_free_path * log(rn)
        '''
        pathl = -self.mfp['Rutherford'] * log(random.random())

        assert (pathl < 1e4), "Path length larger than 10000A: %s > %s" %(pathl, 1e4)
        self.pathl = pathl

        # save path length if we want it
        self.scat_output.addToList('pathl', self.pathl)

    def compute_Eloss(self):
        '''
        energy loss is calculated from Bethe's CSDA
        '''
        assert (self.pathl is not None), "Atempted to compute Eloss when path length is unknown"
        E_loss = self.pathl * bethe_cl_sp(self.m_Z, self.i_energy, self.m_atnd)

        assert (E_loss < self.i_energy), "Energy loss larger than electron energy: %s > %s" %(E_loss, self.i_energy)
        self.E_loss = np.array(E_loss)

        # save energy loss if we want it
        self.scat_output.addToList('E_loss', self.E_loss)

    def compute_sAngles(self):
        self.c2_halfTheta = Rutherford_halfPol(self.i_energy, self.m_Z)

        self.halfPhi = pi*random.random()

        # save scatter type Rutherford if we want it
        self.scat_output.addToList('type', 'Rutherford')

        # save polar angle if we want it
        self.scat_output.addToList('pol_angle', self.c2_halfTheta)

        # save azimuthal angle if we want it
        self.scat_output.addToList('az_angle', self.halfPhi)

# Joy and Luo Bethe as extended from the classical one
class scatter_continuous_JL(scatter_continuous_classical):
    ''' This is the CSLA scattering mode
    Rutherford is the elastic scattering and accounts for angular deviation
    and Joy and Luo modiefied form of Bethe is the continuous energy loss
    '''

    def compute_Eloss(self):
        '''
        energy loss is calculated from Bethe's CSDA
        '''

        assert (self.pathl is not None), "Atempted to compute Eloss when path length is unknown"
        E_loss = self.pathl * bethe_cl_sp(self.m_Z, self.i_energy, self.m_atnd)

        assert (E_loss < self.i_energy), "Energy loss larger than electron energy: %s > %s" %(E_loss, self.i_energy)
        self.E_loss = E_loss

        # save energy loss if we want it
        self.scat_output.addToList('E_loss', self.E_loss)

# explicit shells contributions Bethe as extention from the classical one
class scatter_continuous_explicit(scatter_continuous_classical):
    ''' This is the CSDA scattering mode
    Rutherford is the elastic scattering and accounts for angular deviation
    and the explicit modified version of Bethe is the continuous energy loss
    '''

    def compute_Eloss(self):
        '''
        energy loss is calculated from Bethe's CSDA
        '''

        assert (self.pathl is not None), "Atempted to compute Eloss when path length is unknown"
        E_loss = self.pathl * bethe_cl_sp(self.m_Z, self.i_energy, self.m_atnd)

        assert (E_loss < self.i_energy), "Energy loss larger than electron energy: %s > %s" %(E_loss, self.i_energy)
        self.E_loss = E_loss

        # save energy loss if we want it
        self.scat_output.addToList('E_loss', self.E_loss)


#######################  with units  #########################################
## 1)
class scatter_continuous_classical_wUnits(scatter_continuous_classical):
    def compute_Eloss(self):
        '''
        energy loss is calculated from Bethe's CSDA
        '''

        self.E_loss = self.pathl * bethe_cl_sp(self.m_Z, self.i_energy, self.m_atnd, u_pi_efour)

## 2)
class scatter_continuous_JL_wUnits(scatter_continuous_JL):
    def compute_Eloss(self):
        '''
        energy loss is calculated from Bethe's CSDA
        '''

        self.E_loss = self.pathl * bethe_mod_sp_k(self.m_Z, self.i_energy, self.m_atnd, self.m_k, u_pi_efour)

## 3)
class scatter_continuous_explicit_wUnits(scatter_continuous_explicit):
    def compute_Eloss(self):
        '''
        energy loss is calculated from Bethe's CSDA
        '''

        self.E_loss = self.pathl * bethe_mod_sp(self.i_energy, self.m_atnd, self.m_ns, \
                                    self.m_Es, self.m_nval, self.m_Eval, u_pi_efour)
