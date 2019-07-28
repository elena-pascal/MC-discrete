import numpy as np


from electron import electron
from scattering import scatter_continuous_classical,scatter_continuous_JL, scatter_continuous_explicit
from scattering import scatter_continuous_classical_wUnits, scatter_continuous_JL_wUnits, scatter_continuous_explicit_wUnits
from scattering import scatter_discrete


def scatterOneEl_DS(e_i, material, Emin, Wc, tables_moller, tables_gryz):
    absorbed = False
    scatteredTooLong = False
    num_scatt = 0
    pathl_history = []
    scattAngle_history = []
    type_history = []

    # # first entry
    # # we'll treat this as a scattering with known scattering angle (0)
    # scatter0 = scatter_discrete(e_i, material, Wc, tables_moller, tables_gryz)
    #
    # # calculate path length from total cross section
    # scatter0.compute_pathl()
    # #pathl_history.append(scatter_i.pathl)
    #
    # # update electron position
    # e_i.update_xyz(scatter0.pathl)
    #
    # # determine energy loss and ignore the scatter angle info
    # scatter0.compute_Eloss_sAngles()
    #
    # # update electron energy
    # e_i.update_energy(scatter0.E_loss)


    # now scatter untill absorbed or we decided it went through too many scatterings
    while ((not absorbed) and (not scatteredTooLong)):
        # new instance of scatter
        scatter_i = scatter_discrete(e_i, material, Wc, tables_moller, tables_gryz)
        num_scatt += 1

        # let the electron travel depending on the model used
        scatter_i.compute_pathl()
        #pathl_history.append(scatter_i.pathl)

        # update electron position
        e_i.update_xyz(scatter_i.pathl)

        # check if backscattered
        if (e_i.xyz[2]<= 0.):
            e_i.outcome = 'backscattered'
            #return {'MFP' : np.mean(pathl_history), 'TP' : np.sum(pathl_history), 'num_scatt': num_scatt}
            #return {'num_scatt': num_scatt, 'az_angle': scattAngle_history, 'types': type_history}
            return

        # determine scattering type
        scatter_i.det_type()
        type_history.append(scatter_i.type)

        # determine energy loss and scattering angle
        scatter_i.compute_Eloss_sAngles()
        scattAngle_history.append(scatter_i.c2_halfTheta)

        # update electron energy
        e_i.update_energy(scatter_i.E_loss)

        if (e_i.energy <= Emin):
            absorbed = True
            e_i.outcome = 'absorbed'
            #return {'num_scatt': num_scatt, 'az_angle': scattAngle_history, 'types': type_history}
            return
        # update electron new traveling direction
        e_i.update_direction(scatter_i.c2_halfTheta, scatter_i.halfPhi)

        # if (num_scatt > 1000):
        #     scatteredTooLong = True
        #     e_i.outcome = 'scatteredManyTimes'


    #return {'MFP' : np.mean(pathl_history), 'TP' : np.sum(pathl_history), 'num_scatt': num_scatt}
    #return {'num_scatt': num_scatt, 'az_angle': scattAngle_history, 'types': type_history}
    return

####################### w units #################
def scatterOneEl_DS_wUnits(e_i, material, Emin, Wc, tables_moller, tables_gryz):
    absorbed = False
    scatteredTooLong = False
    num_scatt = 0

    while ((not absorbed) and (not scatteredTooLong)):
        # new instance of scatter
        scatter_i = scatter_discrete_wUnits(e_i, material, Wc, tables_moller, tables_gryz)

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
        if (num_scatt > 1000):
            scatteredTooLong = True
            e_i.outcome = 'scatteredManyTimes'

    return



###############################################################################
################### continuous ################################################
###############################################################################

# 1)
def scatterOneEl_cont_cl(e_i, material, Emin):
    absorbed = False
    scatteredTooLong = False
    num_scatt = 0
    pathl_history = []

    while ((not absorbed) and (not scatteredTooLong)):
        # new instance of scatter
        scatter_i = scatter_continuous_classical(e_i, material)

        # let the electron travel depending on the model used
        scatter_i.compute_pathl()
        pathl_history.append(scatter_i.pathl)

        # update electron position
        e_i.update_xyz(scatter_i.pathl)

        # check if backscattered
        if (e_i.xyz[2]<= 0.):
            e_i.outcome = 'backscattered'
            # exit function here
            return {'MFP' : np.mean(pathl_history), 'TP' : np.sum(pathl_history), 'num_scatt': num_scatt}

        # determine energy loss
        scatter_i.compute_Eloss()

        # update electron energy
        e_i.update_energy(scatter_i.E_loss)

        if (e_i.energy <= float(Emin)):
            absorbed = True
            e_i.outcome = 'absorbed'

        # determine scattering angles
        scatter_i.compute_sAngles()

        # update electron new traveling direction
        e_i.update_direction(scatter_i.c2_halfTheta, scatter_i.halfPhi)

        num_scatt += 1

        # if (num_scatt > 1000):
        #     scatteredTooLong = True
        #     e_i.outcome = 'too far'

    return {'MFP' : np.mean(pathl_history), 'TP' : np.sum(pathl_history), 'num_scatt': num_scatt}

# 2)
def scatterOneEl_cont_JL(e_i, material, Emin):
    absorbed = False
    scatteredTooLong = False
    num_scatt = 0
    pathl_history = []

    while ((not absorbed) and (not scatteredTooLong)):
        # new instance of scatter
        scatter_i = scatter_continuous_JL(e_i, material)

        # let the electron travel depending on the model used
        scatter_i.compute_pathl()
        pathl_history.append(scatter_i.pathl)

        # update electron position
        e_i.update_xyz(scatter_i.pathl)

        # check if backscattered
        if (e_i.xyz[2]<= 0.):
            e_i.outcome = 'backscattered'
            # exit function here
            return {'MFP' : np.mean(pathl_history), 'TP' : np.sum(pathl_history), 'num_scatt': num_scatt}

        # determine energy loss
        scatter_i.compute_Eloss()

        # update electron energy
        e_i.update_energy(scatter_i.E_loss)

        if (e_i.energy <= float(Emin)):
            absorbed = True

        # determine scattering angles
        scatter_i.compute_sAngles()

        # update electron new traveling direction
        e_i.update_direction(scatter_i.c2_halfTheta, scatter_i.halfPhi)

        # num_scatt += 1
        # if (num_scatt > 1000):
        #     scatteredTooLong = True

    return {'MFP' : np.mean(pathl_history), 'TP' : np.sum(pathl_history), 'num_scatt': num_scatt}

# 3)
def scatterOneEl_cont_expl(e_i, material, Emin):
    absorbed = False
    scatteredTooLong = False
    num_scatt = 0
    pathl_history = []

    while ((not absorbed) and (not scatteredTooLong)):
        # new instance of scatter
        scatter_i = scatter_continuous_explicit(e_i, material)

        # let the electron travel depending on the model used
        scatter_i.compute_pathl()
        pathl_history.append(scatter_i.pathl)

        # update electron position
        e_i.update_xyz(scatter_i.pathl)

        # check if backscattered
        if (e_i.xyz[2]<= 0.):
            e_i.outcome = 'backscattered'
            # exit function here
            return {'MFP' : np.mean(pathl_history), 'TP' : np.sum(pathl_history), 'num_scatt': num_scatt}

        # determine energy loss
        scatter_i.compute_Eloss()

        # update electron energy
        e_i.update_energy(scatter_i.E_loss)

        if (e_i.energy <= float(Emin)):
            absorbed = True

        # determine scattering angles
        scatter_i.compute_sAngles()

        # update electron new traveling direction
        e_i.update_direction(scatter_i.c2_halfTheta, scatter_i.halfPhi)

        # num_scatt += 1
        # if (num_scatt > 1000):
        #     scatteredTooLong = True

    return {'MFP' : np.mean(pathl_history), 'TP' : np.sum(pathl_history), 'num_scatt': num_scatt}


 ################ with units ####################

 #1)
def scatterOneEl_cont_cl_wUnits(e_i, material, Emin):
    absorbed = False
    scatteredTooLong = False
    num_scatt = 0

    while ((not absorbed) and (not scatteredTooLong)):
        # new instance of scatter
        scatter_i = scatter_continuous_classical_wUnits(e_i, material)

        # let the electron travel depending on the model used
        scatter_i.compute_pathl()

        # update electron position
        e_i.update_xyz(scatter_i.pathl)

        # check if backscattered
        if (e_i.xyz[2]<= 0.):
            e_i.outcome = 'backscattered'
            return # exit function here

        # determine energy loss
        scatter_i.compute_Eloss()

        # update electron energy
        e_i.update_energy(scatter_i.E_loss)

        if (e_i.energy <= float(Emin)):
            absorbed = True

        # determine scattering angles
        scatter_i.compute_sAngles()

        # update electron new traveling direction
        e_i.update_direction(scatter_i.c2_halfTheta, scatter_i.halfPhi)

        num_scatt += 1
        if (num_scatt > 1000):
            scatteredTooLong = True

    return


# 2)
def scatterOneEl_cont_JL_wUnits(e_i, material, Emin):
    absorbed = False
    scatteredTooLong = False
    num_scatt = 0

    while ((not absorbed) and (not scatteredTooLong)):
        # new instance of scatter
        scatter_i = scatter_continuous_JL_wUnits(e_i, material)

        # let the electron travel depending on the model used
        scatter_i.compute_pathl()

        # update electron position
        e_i.update_xyz(scatter_i.pathl)

        # check if backscattered
        if (e_i.xyz[2]<= 0.):
            e_i.outcome = 'backscattered'
            return # exit function here

        # determine energy loss
        scatter_i.compute_Eloss()

        # update electron energy
        e_i.update_energy(scatter_i.E_loss)

        if (e_i.energy <= float(Emin)):
            absorbed = True

        # determine scattering angles
        scatter_i.compute_sAngles()

        # update electron new traveling direction
        e_i.update_direction(scatter_i.c2_halfTheta, scatter_i.halfPhi)

        num_scatt += 1
        if (num_scatt > 1000):
            scatteredTooLong = True

    return


 # 3)
def scatterOneEl_cont_expl_wUnits(e_i, material, Emin):
    absorbed = False
    scatteredTooLong = False
    num_scatt = 0

    while ((not absorbed) and (not scatteredTooLong)):
        # new instance of scatter
        scatter_i = scatter_continuous_explicit_wUnits(e_i, material)

        # let the electron travel depending on the model used
        scatter_i.compute_pathl()

        # update electron position
        e_i.update_xyz(scatter_i.pathl)

        # check if backscattered
        if (e_i.xyz[2]<= 0.):
            e_i.outcome = 'backscattered'
            return # exit function here

        # determine energy loss
        scatter_i.compute_Eloss()

        # update electron energy
        e_i.update_energy(scatter_i.E_loss)

        if (e_i.energy <= float(Emin)):
            absorbed = True

        # determine scattering angles
        scatter_i.compute_sAngles()

        # update electron new traveling direction
        e_i.update_direction(scatter_i.c2_halfTheta, scatter_i.halfPhi)

        num_scatt += 1
        if (num_scatt > 1000):
            scatteredTooLong = True

    return
