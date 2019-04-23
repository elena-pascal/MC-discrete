from scimath.units.energy import J, eV, KeV
from scimath.units.api import UnitScalar, UnitArray, convert, has_units

from scipy.constants import pi, Avogadro, hbar, m_e, e, epsilon_0, eV

def exp_params():
    params = {'file': 'parameters.py'}

    # incident energy
    params['E'] = UnitScalar(20000, units="eV")

    # cut off energy value for free electron scattering energy transfer [5..100]
    params['Ec'] = UnitScalar(10., units="eV")

    return params



class electron:
    ''' An electron is an object with defined energy position and
        direction in the sample frame.
        Input can have units.
    '''
    def __init__(self, energy=[], position=[], direction=[]):
        self.energy = energy
        self.xyz = position
        self.dir = direction


    def update_energy(self, energyLoss):
        ''' update electron after every scattering
            keep record of the history by appending new iformation to lists
        '''

        newEnergy = self.energy[-1] - energyLoss
        self.energy.append(newEnergy)


    def update_xyz(self, pathLength):
        newPosition = self.xyz[-1] + pathLength * self.dir[-1]
        self.xyz.append(newPosition)

    def update_drection(self, c2_halfPhi, halfTheta):
        shphi = (1. - c2_halfPhi)**0.5
        chphi = c2_halfPhi**0.5
        shpsi = sin(halfTheta)
        chps = (1. - shpsi)**0.5
        newDirectin = newdir(shphi, chphi, shpsi, chpsi, self.dir[-1])


    def totalPathLenght(self):
        return sum(self.xyz)
