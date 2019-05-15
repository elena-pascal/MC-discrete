import numpy as np
from math import acos

from scimath.units.energy import J, eV, KeV
from scimath.units.length import m, cm, km, angstrom
from scimath.units.api import UnitScalar, UnitArray, convert, has_units


from electron import electron
from material import material
from scattering import scatter_discrete, scatter_continuous
from extFunctions import gryz_dCS, moller_dCS
from integrals import trapez_table, extF_limits_gryz, extF_limits_moller
from parameters import c_pi_efour

material = material('Al')

num_el = 200
E0 = UnitScalar(30000, units = 'eV') # eV
Emin = UnitScalar(5000, units = 'eV') # eV
tilt = 70. # degrees
pos0 = np.array([0., 0., 0.,])
dir0 = np.array([0., -np.sin(np.radians(tilt)) , np.cos(np.radians(tilt))])
#mode = 'DS' # discrete scattering
mode = 'DS'

nBinsW = 1000
nBinsE = 200

Wc = UnitScalar(5, units = 'eV')

def u2n(value_with_units):
    '''
    Tranforms quantity with units in
    numpy unitless array
    '''
    return np.array(value_with_units)





BSEcount = 0
BSEs = []
position = []
theta_R = []
phi_R = []
theta_G = []
phi_G = []
theta_M = []
phi_M = []

if (mode == 'DS'):
    print '---- calculating Moller tables'
    # remove the function dependece on the constant and get rid of units
    funcToint_M = lambda E, W, n_e : u2n(moller_dCS(E, W, n_e, c_pi_efour))

    tables_moller = trapez_table( float(E0), float(Emin), np.array([Wc]), float(material.fermi_e()), \
                np.array([material.nval()]), funcToint_M, nBinsW, nBinsE)
    #print tables_moller[2][0, 1, :]
    #print tables_moller[3][0, 1, :]
    print


    print '---- calculating Gryzinski tables'
    tables_gryz = []

    funcToint_G = lambda E, W, n_e, Ebi : u2n(gryz_dCS(E, W, n_e,\
                                                c_pi_efour, Ebi))

    tables_gryz = trapez_table(float(E0), float(Emin), np.array(material.Es()), float(material.fermi_e()),\
                    np.array(material.ns()), funcToint_G, nBinsW, nBinsE )
    #print tables_gryz[2][1, 1, :]
    #print tables_gryz[3][1, 1, :]
    print


    print 'starting discrete inelastic scattering'
    for i in range(num_el):
        if (i% 100 == 0):
            print '-------- starting electron:', i
        e_i = electron(E0, pos0, dir0)

        position.append(pos0)
        backscattered = False
        absorbed = False
        scatteredTooLong = False
        num_scatt = 0
        while ((not backscattered) and (not absorbed) and (not scatteredTooLong)):

            # new instance of scatter
            scatter_i = scatter_discrete(e_i, material, Wc, tables_moller, tables_gryz)

            # let the electron travel depending on the model used
            scatter_i.compute_pathl()

            # update electron position
            e_i.update_xyz(scatter_i.pathl)

            # check if backscattered
            if (e_i.xyz[2]<= 0.):
                backscattered = True

                BSEs.append(float(e_i.energy))
                BSEcount += 1

            # determine scattering type
            scatter_i.det_type()

            if (scatter_i.type == 'Rutherford'):
                # determine scattering angles
                scatter_i.compute_sAngles()

                phi_R.append(2.*scatter_i.halfPhi)
                theta_R.append(2.*acos(scatter_i.c2_halfTheta**0.5))

                # update electron new traveling direction
                e_i.update_direction(scatter_i.c2_halfTheta, scatter_i.halfPhi)

            elif('Gryzinski' in scatter_i.type):
                # determine energy loss
                scatter_i.compute_Eloss()

                # determine scattering angles
                scatter_i.compute_sAngles()

                phi_G.append(2.*scatter_i.halfPhi)
                theta_G.append(2.*acos(scatter_i.c2_halfTheta**0.5))

                # update electron energy
                e_i.update_energy(scatter_i.E_loss)

                if (e_i.energy <= float(Emin)):
                    absorbed = True

                # update electron new traveling direction
                e_i.update_direction(scatter_i.c2_halfTheta, scatter_i.halfPhi)

            elif(scatter_i.type == 'Moller'):
                # determine energy loss
                scatter_i.compute_Eloss()

                # determine scattering angles
                scatter_i.compute_sAngles()

                phi_M.append(2.*scatter_i.halfPhi)
                theta_M.append(2.*acos(scatter_i.c2_halfTheta**0.5))

                # update electron energy
                e_i.update_energy(scatter_i.E_loss)

                if (e_i.energy <= float(Emin)):
                    absorbed = True

                # update electron new traveling direction
                e_i.update_direction(scatter_i.c2_halfTheta, scatter_i.halfPhi)

            elif (scatter_i.type == 'Quinn'):
                # determine energy loss
                scatter_i.compute_Eloss()

                # update electron energy
                e_i.update_energy(scatter_i.E_loss)

                if (e_i.energy <= Emin):
                    absorbed = True

            num_scatt += 1
            if (num_scatt > 10000):
                scatteredTooLong = True

        # append the position history of this electron
        position.append(e_i.xyz_hist)

elif (mode == 'continuous'):
    print 'starting continuous slowing down inelastic scattering'
    for i in range(num_el):
        if (i% 100 == 0):
            print '-------- starting electron:', i
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


print 'Total BSE electrons', BSEcount






fileBSE = 'BSE_70tilt_ds.out'
with open(fileBSE, 'w') as f:
    for item in BSEs:
        f.write("%s\n" % item)

print 'BSE data was written to', fileBSE


file_pos = 'positions_70tilt_ds.out'
with open(file_pos, 'w') as f:
        for item in position:
            f.write("%s\n" % item)

print 'pos data was written to', file_pos



file_phiR = 'phiR_70tilt_ds.out'
with open(file_phiR, 'w') as f:
        for item in phi_R:
            f.write("%s\n" % item)

print 'phi angles for Rutherford was written to', file_phiR


file_thetaR = 'thetaR_70tilt_ds.out'
with open(file_thetaR, 'w') as f:
        for item in theta_R:
            f.write("%s\n" % item)

print 'theta angles for Rutherford was written to', file_thetaR

#
# file_phiG = 'phiG_50tilt.out'
# with open(file_phiG, 'w') as f:
#         for item in phi_G:
#             f.write("%s\n" % item)
#
# print 'phi angles for Gryzinski was written to', file_phiG
#
# file_thetaG = 'thetaG_50tilt.out'
# with open(file_thetaG, 'w') as f:
#         for item in theta_G:
#             f.write("%s\n" % item)
#
# print 'theta angles for Gryzinski was written to', file_thetaG
#
# file_phiM = 'phiM_50tilt.out'
# with open(file_phiM, 'w') as f:
#         for item in phi_M:
#             f.write("%s\n" % item)
#
# print 'phi angles for Moller was written to', file_phiM
#
# file_thetaM = 'thetaM_50tilt.out'
# with open(file_thetaM, 'w') as f:
#         for item in theta_M:
#             f.write("%s\n" % item)
#
# print 'theta angles for Moller was written to', file_thetaM
