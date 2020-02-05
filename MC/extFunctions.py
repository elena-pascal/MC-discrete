#from scimath.units.api import has_units
import numpy as np
import numpy.ma as ma
from scipy import stats, integrate
import sys
import os.path as path

from MC.errors import E_lossTooLarge
from MC.parameters import pi_efour


###################################################################
#                       Excitation functions d_sigma/dW           #
###################################################################
# Probability that an electron of energy E will loose energy W
# in a scattering event

# Since these functions are only used for determining the energy loss
# in the numerical integration, I disabled the units for now
# TODO: do this nicer though

# 1b) Mott numerical differential cross sections
class mottTable:
    '''
    Mott cross section table from ioffe.ru/ES/Elastic

    sigma (cm^2) is a function of (E, Z)

    dSigma/dOmega (cm^2/str) is a function (E, Z, theta)

    E = [5 eV, 30 keV]

    theta = [0, 180]

    Functions:
        readIoffe: read the information from the database

        toProbTable : calculate CDF table from dCS

        mapToMemory: save the CDF table to a memory map


    '''
    def __init__(self, Z, name):
        self.Z = Z
        self.path = '../tables/mott/'
        self.target = path.join(self.path, name + '.table')
        self.name = name

        self.probTable = None

        ioffeFile = path.join(self.path+ 'ioffe/' + name + '.ioffe')

        # check existence
        if (ioffeFile):
            self.ioffeFile = ioffeFile
        else:
            sys.exit('Data for %s was not found' %name)

        with open(self.ioffeFile, 'r') as file:
            lines = file.readlines()
            # check if this is what we expect
            if (int(lines[2].split()[-1]) is not self.Z):
                sys.exit('Trying to read file %s but expecting Z:%s' %(self.ioffeFile, self.Z))

            self.sizeTheta = int(lines[5].split()[0])
            self.sizeE = int(lines[6].split()[0])

        self.Es = np.empty(self.sizeE)

        # initialise the sigma(E) array
        self.sigmas = np.empty(self.sizeE)

        # initialise the dCS(E, theta) array
        self.dCS = np.empty([self.sizeE, self.sizeTheta])

    def readIoffe(self):
        '''
        read the Ioffe file and populate the sigma and dCS arrays
        '''
        with open(self.ioffeFile, 'r') as file:
            lines = file.readlines()

            self.thetas = np.array([float(item) for item in lines[12].strip('theta [grad]=').split()])

            for index, line in enumerate(lines[14:-1]):
                items = line.replace('|', '').split()
                self.Es[index] = float(items[0])

                self.sigmas[index] = float(items[-1])

                self.dCS[index] = np.array([float(item) for item in items[1:-1]])

        # The data in the ioffe table is descending in energy
        # Since we use bisect it's useful to keep all the values in
        # ascending order. So we flip along E all arrays
        self.Es     = np.flip(self.Es)
        self.sigmas = np.flip(self.sigmas)
        self.dCS    = np.flip(self.dCS, axis=0)

    def toProbTable(self):
        '''
        Transform the dCS table to a probability (CDF) table

        i.e. integrate (sum) over theta
        '''
        # cumulative sum on the stepwise integral
        cumInt = np.cumsum(self.dCS, axis=1)

        # divide by total sum and populate probTable
        self.probTable = cumInt/np.broadcast_to(np.sum(self.dCS, axis=1), cumInt.T.shape).T


    def mapToMemory(self):
        '''
        Put the dCS values in a memory map
        '''
        # create a memory map with defined dtype and shape
        storedMap = np.memmap(filename = self.target,
                       dtype    = 'float32',
                       mode     = 'w+',
                       shape    = (self.sizeE, self.sizeTheta)   )

        # fill allocated memory with table
        storedMap[:] = self.probTable[:]

        # flush to memory
        del storedMap

        print ('Table written to target:', self.target)


    def readFromMemory(self):
        '''
        Load the dCS table from memory map
        '''
        if self.target:
            # read the table from memory
            self.probTable =  np.memmap(filename = self.target,
                           dtype    = 'float32',
                           mode     = 'r',
                           shape    = (self.sizeE, self.sizeTheta)   )
            print ('read Mott table from memory for %s\n' %self.name)
        else:
            # generate this table
            self.generate()


    def generate(self):
        '''
        generate the memory map for this material
        '''
        self.readIoffe()
        self.toProbTable()
        self.mapToMemory()



# 2b) Moller free electron discrete cross section
#@has_units
def moller_dCS(E, W, nfree, c_pi_efour=pi_efour):
    """ Calculate the Moller inelastic discrete cross section

        Parameters
        ----------
        E      : array : units = eV
                       incident energy

        W      : array : units = eV
                       energy loss

        nfree  : array : units = dim

        c_pi_efour: scalar: units = cm**2 * eV**2

        Returns
        -------
        dCS  : array : units = cm**2
    """
    eps = W/E

    dCS = np.where(eps<1., nfree*c_pi_efour *( 1./(eps**2) +
                  ( 1./((1.-eps)**2) ) - ( 1./(eps*(1.-eps)) ) )/ E**3,
                  0)

    return dCS


# 2b) Gryzinski differential cross section for core shell electrons
#@has_units
def gryz_dCS(E, W, nsi, Ebi, c_pi_efour=pi_efour):
    """ Calculate the Moller inelastic cross section

        Parameters
        ----------
        E      : array : units = eV
                       incident energy

        W      : array : units = eV
                       energy loss

        Ebi    : array : units = eV
                       binding energy of shell i

        nsi    : array : units = dim
                       number of electrons in shell i

        c_pi_efour: scalar: units = cm**2 * eV**2

        Returns
        -------
        dCS    : array : units = cm**2
    """

    eps = W/E
    epsB = Ebi/E

    dCS = np.where((eps<1.)&(epsB<1.), nsi * c_pi_efour * eps * (1. + epsB)**(-1.5) * (1. - eps)**(epsB/(epsB+eps)) * ((1. - epsB) +\
                                        4. * epsB * np.log(2.7 + ((1. - eps)/epsB)**(0.5) )/(3.*eps) )   /( W**3), \
                                        0)
    return dCS

# 2b') Gryzinski differential cross section for core shell electrons
# but following Patrik's approach where he sums up all the shells cotributions
#@has_units
def gryz_dCS_P(E, W, nsi, Ebi, c_pi_efour=pi_efour):
    """ Calculate the Moller inelastic cross section

        Parameters
        ----------
        E      : array : units = eV
                       incident energy

        W      : array : units = eV
                       energy loss

        Ebi    : array : units = eV
                       array of binding energy of shell i

        nsi    : array : units = dim
                       array of number of electrons per shell i

        c_pi_efour: scalar: units = cm**2 * eV**2

        Returns
        -------
        dCS    : array : units = cm**2
    """

    dCS = 0.

    for indx, ni in enumerate(nsi):
        if  (W > E) or (Ebi[indx] > E):
            dCS += 0.

        else:
            eps = W*1./E
            epsB = Ebi[indx]*1./E
            dCS += ni * c_pi_efour * eps * (1. + epsB)**(-1.5) * (1. - eps)**(epsB/(epsB+eps)) * ((1. - epsB) +
                                       4. * epsB * log(2.7 + ((1. - eps)/epsB)**(0.5) )/(3.*eps) )   /( W**3)


    return dCS

# 2c) Dielectric function proposed by Powell (1985) for the optical dielectric limit
# see Powell 'Calculations of electron inelastic mean free paths from experimental optical data'
# Surface and and interface analysis, 7(6): 263-274, 1985
#@has_units
# def diel_Pow_dCS(E, W, eps_W, powell_c, c_me=me, c_e=e, c_hbar=hbar):
#     """ Calculate the Powell formula for the dielectric inelastic cross section
#
#         Parameters
#         ----------
#         E      : array : units = eV
#                        incident energy
#
#         W      : array : units = eV
#                        energy loss
#
#         eps_W  : array : units = dim
#                         dielectric function evaulated at W
#
#         Penn_b : array : units = dim
#                         Powell c parameter
#
#         c_me   : array : units = kg
#                        electron mass constant
#
#         c_e    : array : units = coulomb
#                        electron charge constant
#
#         c_hbar : array : units =
#                        reduced Plank's constant
#
#         Returns
#         -------
#         dCS    : array : units = cm**2
#     """
#
#     try:
#         dCS = c_me * c_e**2 * eps_W*ln(powell_c * E/ W) / (2. * pi * c_hbar**2 * E)
#
#         if  (1. - eps < 0):
#             raise E_lossTooLarge
#
#     except E_lossTooLarge:
#         print ' The energy loss is larger than the current electron energy in Powell formulation of the electrostatic discrete CS'
#         print ' W is', W ,'and E is', E
#         print ' Stopping'
#         sys.exit()
#
#     return dCS
