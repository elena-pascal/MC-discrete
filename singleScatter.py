import numpy as np
from math import acos


from scimath.units.energy import J, eV, KeV
from scimath.units.length import m, cm, km, angstrom
from scimath.units.api import UnitScalar, UnitArray, convert, has_units


from electron import electron
from scattering import scatter_discrete, scatter_continuous


def singleScatter_DS(e_i, material, Emin, Wc, tables_moller, tables_gryz):
    absorbed = False
    scatteredTooLong = False
    num_scatt = 0

    while ((not absorbed) and (not scatteredTooLong)):
        # new instance of scatter
        scatter_i = scatter_discrete(e_i, material, Wc, tables_moller, tables_gryz)

        # let the electron travel depending on the model used
        scatter_i.compute_pathl()

        # update electron position
        e_i.update_xyz(scatter_i.pathl)

        # check if backscattered
        if (e_i.xyz[2]<= 0.):
            e_i.outcome = 'backscattered'
            return # exit function here

        # determine scattering type
        scatter_i.det_type()

        if (scatter_i.type == 'Rutherford'):
            # determine scattering angles
            scatter_i.compute_sAngles()

            #phi_R.append(2.*scatter_i.halfPhi)
            #theta_R.append(2.*acos(scatter_i.c2_halfTheta**0.5))

            # update electron new traveling direction
            e_i.update_direction(scatter_i.c2_halfTheta, scatter_i.halfPhi)

        elif('Gryzinski' in scatter_i.type):
            # determine energy loss
            scatter_i.compute_Eloss()

            # determine scattering angles
            scatter_i.compute_sAngles()

            #phi_G.append(2.*scatter_i.halfPhi)
            #theta_G.append(2.*acos(scatter_i.c2_halfTheta**0.5))

            # update electron energy
            e_i.update_energy(scatter_i.E_loss)

            if (e_i.energy <= float(Emin)):
                absorbed = True
                e_i.outcome = 'absorbed'

            # update electron new traveling direction
            e_i.update_direction(scatter_i.c2_halfTheta, scatter_i.halfPhi)

        elif(scatter_i.type == 'Moller'):
            # determine energy loss
            scatter_i.compute_Eloss()

            # determine scattering angles
            scatter_i.compute_sAngles()

            #phi_M.append(2.*scatter_i.halfPhi)
            #theta_M.append(2.*acos(scatter_i.c2_halfTheta**0.5))

            # update electron energy
            e_i.update_energy(scatter_i.E_loss)

            if (e_i.energy <= float(Emin)):
                absorbed = True
                e_i.outcome = 'absorbed'

            # update electron new traveling direction
            e_i.update_direction(scatter_i.c2_halfTheta, scatter_i.halfPhi)

        elif (scatter_i.type == 'Quinn'):
            # determine energy loss
            scatter_i.compute_Eloss()

            # update electron energy
            e_i.update_energy(scatter_i.E_loss)

            if (e_i.energy <= Emin):
                absorbed = True
                e_i.outcome = 'absorbed'

        num_scatt += 1
        if (num_scatt > 500):
            scatteredTooLong = True
            e_i.outcome = 'scatteredManyTimes'

    return




def singleScatter_cont (material, E0, Emin, tilt):
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
