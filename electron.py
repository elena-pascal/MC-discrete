from rotation import newdir, newdircos_oldMC

from math import sin, cos
import numpy as np


class electron:
    ''' An electron is an object with defined energy position and
        direction in the sample frame.
        Input can have units.
    '''
    def __init__(self, energy, position, direction):
        self.energy = energy
        self.xyz    = position
        self.dir    = direction

        self.outcome = None

        self.y_local = np.array([0., 1., 0.]) # local coordinate system

        #self.energy_hist = []
        #self.xyz_hist = []
        #self.dir_hist = []


    def update_energy(self, energyLoss):
        ''' update electron after every scattering
            keep record of the history by appending new information to lists
        '''
        energy = self.energy - energyLoss
        if energy:
            self.energy = energy
        else:
            print ('------- None value in energy found when updating energy')
        #self.energy_hist.append(newEnergy)

    def update_xyz(self, pathLength):
        newPosition = self.xyz + pathLength*self.dir
        self.xyz = newPosition
        #self.xyz_hist.append(newPosition)


    def update_direction(self, c2_halfTheta, halfPhi):
        s_hTheta = (1. - c2_halfTheta)**0.5 # sin(halfTheta) is positive on [0, pi)
        c_hTheta = c2_halfTheta**0.5 # halfTheta = [0, pi/2], so cos(halfTheta) is positive
        s_hPhi = sin(halfPhi)
        c_hPhi = (1. - s_hPhi**2)**0.5 # halfPhi = [0, pi] so cos(halfPhi) is positive

        newDirection_andy = newdir(s_hTheta, c_hTheta, s_hPhi, c_hPhi, self.y_local, self.dir)
        # after many scattering events d will lose its normalisation due to precision limitations,
        # so it's good to renormalise
        (self.dir, self.y_local) = [(dir/ np.linalg.norm(dir)) for dir in newDirection_andy]
    #    print 'y local', self.y_local



        #self.dir_hist.append(newDirection)
        #
        # c_Theta = 2.*c2_halfTheta - 1.
        # s_Theta = (1. - c_Theta**2)**0.5
        # s_Phi = sin(2.*halfPhi)
        # c_Phi = (1. - s_Phi**2)**0.5
        #
        # newDir = newdircos_oldMC(s_Theta, c_Theta, s_Phi, c_Phi, self.dir)
        # self.dir = newDir/np.linalg.norm(newDir)


    def totalPathLenght(self):
        return sum(self.xyz)
