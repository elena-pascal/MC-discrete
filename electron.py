from scimath.units.energy import J, eV, KeV
from scimath.units.api import UnitScalar, UnitArray, convert, has_units

from scipy.constants import pi, Avogadro, hbar, m_e, e, epsilon_0, eV
from rotation import newdir

from math import sin, cos
import numpy as np

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
    def __init__(self, energy, position, direction):
        self.energy = energy
        self.xyz = position
        self.dir = direction

        self.y_local = np.array([0., 1., 0.]) # local coordinate system

        self.energy_hist = []
        self.xyz_hist = []
        self.dir_hist = []


    def update_energy(self, energyLoss):
        ''' update electron after every scattering
            keep record of the history by appending new iformation to lists
        '''
        self.energy = self.energy - energyLoss
        #self.energy_hist.append(newEnergy)

    def update_xyz(self, pathLength):
        newPosition = self.xyz + float(pathLength) * self.dir
        self.xyz = newPosition
        #self.xyz_hist.append(newPosition)

    def update_direction(self, c2_halfTheta, halfPhi):
        s_hTheta = (1. - c2_halfTheta)**0.5 # sin(halfTheta) should be positive see below
        c_hTheta = c2_halfTheta**0.5 # halfTheta = [0, pi/2], so cos(halfTheta) is positive
        s_hPhi = sin(halfPhi)
        c_hPhi = (1. - s_hPhi**2)**0.5 # hPhi = [0, pi] so cos(hPhi) is positive

        newDirection_andy = newdir(s_hTheta, c_hTheta, s_hPhi, c_hPhi, self.y_local, self.dir)
        # after many scattering events d will lose its normalisation due to precision limitations,
        # so it's good to renormalise
        (self.dir, self.y_local) = [(dir/ np.linalg.norm(dir)) for dir in newDirection_andy]
        #self.dir_hist.append(newDirection)

    def totalPathLenght(self):
        return sum(self.xyz)
