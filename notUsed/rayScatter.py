import ray
import numpy as np
from math import acos

from singleScatter import scatter_continuous, scatter_discrete
from electron import electron



@ray.remote
def rayScatter_cont(material, num_el, E0, Emin, tilt):
    pos0 = np.array([0., 0., 0.,])
    dir0 = np.array([0., -np.sin(np.radians(tilt)) , np.cos(np.radians(tilt))])

    BSEcount = 0
    BSEs = []
    position = []
    theta_R = []
    phi_R = []

    e_i = electron(E0, pos0, dir0)

    backscattered = False
    absorbed = False
    scatteredTooLong = False
    num_scatt = 0

    while ((not backscattered) and (not absorbed) and (not scatteredTooLong)):

        # new instance of scatter
        scatter_i = scatter_continuous(e_i, material)

        # let the electron travel depending on the model used
        scatter_i.compute_pathl()

        # update electron position
        e_i.update_xyz(scatter_i.pathl)

        # check if backscattered
        if (e_i.xyz[2]<= 0.):
            backscattered = True

            BSEs.append(float(e_i.energy))
            BSEcount += 1

        # determine energy loss
        scatter_i.compute_Eloss()

        # update electron energy
        e_i.update_energy(scatter_i.E_loss)

        if (e_i.energy <= float(Emin)):
            absorbed = True

        # determine scattering angles
        scatter_i.compute_sAngles()

        phi_R.append(2.*scatter_i.halfPhi)
        theta_R.append(2.*acos(scatter_i.c2_halfTheta**0.5))

        # update electron new traveling direction
        e_i.update_direction(scatter_i.c2_halfTheta, scatter_i.halfPhi)

        num_scatt += 1
        if (num_scatt > 10000):
            scatteredTooLong = True

    # append the position history of this electron
    position.append(e_i.xyz_hist)
