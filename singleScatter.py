import numpy as np
from electron import electron
from scattering import scatter

material = material('Al')

num_el = 100
E0 = 20000 # eV
tilt = 60 # degrees
pos0 = [0., 0., 0.,]
dir0 = [-np.sin(np.radians(tilt)), 0. , np.cos(np.radians(tilt))]
model = 'DS' # discrete scattering


for i in num_el:
    ei = electron(E, pos0, dir0)
    while ((electron.xyz[2]>=0.) and (electron.energy>=10000)):# not backscattered
        scatter = scatter(electron, material)

        # let the electron travel depending on the model used
        scatter.compute_patl()
        print 'Path length is:', scatter.pathl

        # update electron position
        electron.update_xyz(scatter.pathl)

        # determine scattering type
        scatter.det_type()
        print 'Scatter type is:', scatter.type

        # determine energy loss
        scatter.compute_Eloss()
        print 'Energy loss is:', scatter.E_loss

        # update electron energy
        electron.update_energy(scatter.E_loss)

        # determine scattering angles
        scatter.calculate_sAngles()
        print 'cos square half phi is:', scatter.c2_halfPhi
        print 'half theta is:', scatter.halfTheta

        # update electron new traveling direction
        electron.update_dir(scatter.c2_halfPhi, scatter.halfTheta)
