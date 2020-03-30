from MC.rotation import newdir, newdircos_oldMC
from MC.fileTools import thingsToSave
from math import sin, cos
import numpy as np


class electron:
    ''' An electron is an object with defined energy position and
        direction in the sample frame.
        Input can have units.
    '''
    def __init__(self, energy, Emin, position, direction, outList):
        '''
        energy   : energy of incident electron in eV

        Emin     : minimum energy untill for which the electron is tracked, in eV

        position : entry location in sample ref frame

        direction: direction of incidence in sample ref frame

        outList  : {'el_output'  : [list of electron parameters to save],
                    'scat_output': [list of scattering parameter to save]}
        '''
        self.energy = energy
        self.Emin   = Emin
        self.xyz    = position
        self.dir    = direction

        self.outcome = None

        self.diffracting = False

        self.y_local = np.array([0., 1., 0.]) # local coordinate system

        # object of list of parameters to save
        self.el_output = thingsToSave(outList['el_output'])
        self.scat_output = thingsToSave(outList['scat_output'])

    def update_energy(self, energyLoss):
        ''' update electron after every scattering
            keep record of the history by appending new information to lists
        '''
        energy = self.energy - energyLoss
        assert energy, 'None value in energy found when updating energy'

        # check if absorbed
        if (energy <= self.Emin):
            self.outcome = 'abs'
            self.saveOutcomes()

        self.energy = energy

        # save energy if we want it
        self.scat_output.addToList('energy', self.energy)

    def changeDiffState(self, diff):
        self.diffracting = diff

    def update_xyz(self, pathLength, zmax):
        newPosition = self.xyz + pathLength*self.dir

        # check if backscattered
        if (newPosition[2] <= 0.):
            self.outcome = 'bks'
            self.saveOutcomes()

        # check if transmited thorough
        elif (newPosition[2] >= zmax):
            # if last scatter event was diffracting out then we have a TSE
            if self.diffracting:
                self.outcome = 'trdf'
            else:
                self.outcome = 'trsm'

            self.saveOutcomes()

        else:
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


    def totalPathLenght(self):
        return sum(self.xyz)

    def saveOutcomes(self):
        ''' save all the electron parameters we are interested in'''
        self.el_output.addToList('outcome', self.outcome)

        self.el_output.addToList('final_E', self.energy)

        self.el_output.addToList('final_dir', self.dir)

        self.el_output.addToList('last_pos', self.xyz)

        self.scat_output.addToList('position', self.xyz)





###############################################################################
