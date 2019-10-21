from rotation import newdir, newdircos_oldMC

from math import sin, cos
import numpy as np


class electron:
    ''' An electron is an object with defined energy position and
        direction in the sample frame.
        Input can have units.
    '''
    def __init__(self, energy, Emin, position, direction, thingsToSave):
        self.energy = energy
        self.Emin   = Emin
        self.xyz    = position
        self.dir    = direction

        self.outcome = None

        self.y_local = np.array([0., 1., 0.]) # local coordinate system

        # object of list of parameters to save
        self.el_output = thingsToSave['el_output']
        self.scat_output = thingsToSave['scat_output']

    def update_energy(self, energyLoss):
        ''' update electron after every scattering
            keep record of the history by appending new information to lists
        '''
        energy = self.energy - energyLoss
        assert energy, 'None value in energy found when updating energy'

        # check if absorbed
        if (energy <= self.Emin):
            self.outcome = 'absorbed'
            self.saveOutcomes()

        self.energy = energy

        # save energy if we want it
        self.scat_output.addToList('energy', self.energy)

    def update_xyz(self, pathLength):
        newPosition = self.xyz + pathLength*self.dir

        # check if backscattered
        if (newPosition[2] <= 0.):
            self.outcome = 'backscattered'
            self.saveOutcomes()

        self.xyz = newPosition

        # save position if we want it
        self.scat_output.addToList('position', self.xyz)

    def update_direction(self, c2_halfTheta, halfPhi):
        s_hTheta = (1. - c2_halfTheta)**0.5 # sin(halfTheta) is positive on [0, pi)
        c_hTheta = c2_halfTheta**0.5 # halfTheta = [0, pi/2], so cos(halfTheta) is positive
        s_hPhi = sin(halfPhi)
        c_hPhi = (1. - s_hPhi**2)**0.5 # halfPhi = [0, pi] so cos(halfPhi) is positive

        newDirection_andy = newdir(s_hTheta, c_hTheta, s_hPhi, c_hPhi, self.y_local, self.dir)
        # after many scattering events d will lose its normalisation due to precision limitations,
        # so it's good to renormalise
        (self.dir, self.y_local) = [(dir/ np.linalg.norm(dir)) for dir in newDirection_andy]


        # save direction if we want it
        self.scat_output.addToList('direction', self.dir)

        # TODO: test
        # c_Theta = 2.*c2_halfTheta - 1.
        # s_Theta = (1. - c_Theta**2)**0.5
        # s_Phi = sin(2.*halfPhi)
        # c_Phi = (1. - s_Phi**2)**0.5
        #
        # newDir = newdircos_oldMC(s_Theta, c_Theta, s_Phi, c_Phi, self.dir)
        # self.dir = newDir/np.linalg.norm(newDir)


    def totalPathLenght(self):
        return sum(self.xyz)

    def saveOutcomes(self):
        ''' save all the electron parameters we are interested in'''

        self.el_output.addToList('outcome', self.outcome)

        self.el_output.addToList('final_E', self.energy)

        self.el_output.addToList('final_dir', self.dir)


###############################################################################
from scattering import scatter_discrete

def trajectory_DS(electron, E_i, material, Wc, maxScatt, tables):
    ''' follow a full electron trajectory'''

    num_scatt = 0

    while ((electron.outcome is not 'absorbed') and (electron.outcome is not 'backscattered')): # not backscattered nor absorbed nor scattered too long
        # new instance of scatter
        scatter = scatter_discrete(electron, material, Wc, tables)

        num_scatt += 1
        if (num_scatt > maxScatt):
             electron.outcome = 'scatteredTooLong'
             electron.saveOutcomes()
             return # exit while loop

        # let the electron travel depending on the model used
        scatter.compute_pathl()

        # update electron position
        electron.update_xyz(scatter.pathl)

        # determine scattering type
        scatter.det_type()

        # determine energy loss and scattering angle
        scatter.compute_Eloss_and_angles()

        # update electron energy
        electron.update_energy(scatter.E_loss)

        # update electron new traveling direction
        electron.update_direction(scatter.c2_halfTheta, scatter.halfPhi)
