import numpy as np
from math import acos
from scimath.units.energy import J, eV, KeV
from scimath.units.length import m, cm, km, angstrom
from scimath.units.api import UnitScalar, UnitArray, convert, has_units


from electron import electron
from material import material
from scattering import scatter
from extFunctions import gryz_dCS, moller_dCS
from integrals import trapez_table, extF_limits_gryz, extF_limits_moller
from parameters import c_pi_efour

material = material('Al')

num_el = 100
E0 = UnitScalar(20000, units = 'eV') # eV
Emin = UnitScalar(9000, units = 'eV') # eV
tilt = 50. # degrees
pos0 = np.array([0., 0., 0.,])
dir0 = np.array([0., -np.sin(np.radians(tilt)) , np.cos(np.radians(tilt))])
model = 'DS' # discrete scattering

nBinsW = 100
nBinsE = 10

Wc = UnitScalar(100, units = 'eV')

def u2n(value_with_units):
    '''
    Tranforms quantity with units in
    numpy unitless array
    '''
    return np.array(value_with_units)


print '---- calculating Moller tables'
a_M, b_M = u2n(extF_limits_moller(E0, Wc))

# remove the function dependece on the constant and get rid of units
funcToint_M = lambda E, W, n_e : u2n(moller_dCS(E, W, material.get_nval(), c_pi_efour))

tables_moller = trapez_table(a_M, b_M, np.array(Emin), np.array(E0), material.get_nval(), \
            funcToint_M, nBinsW, nBinsE)
#print tables_moller


print '---- calculating Gryzinski tables'
tables_gryz = []
for ishell in range(len(material.get_ns())):
    a_G, b_G = u2n(extF_limits_gryz(E0, material.get_Es()[ishell]))
    funcToint_G = lambda E, W, n_e, Ebi : u2n(gryz_dCS(E, W, material.get_ns()[ishell],\
                                            c_pi_efour, material.get_Es()[ishell]))

    tables_gryz.append(trapez_table(a_G, b_G, np.array(Emin), np.array(E0),\
                material.get_ns()[ishell], funcToint_G, nBinsW, nBinsE,  material.get_Es()[ishell]) )
#print tables_gryz

count = 0
BSE = []
position = []
theta_R = []
phi_R = []
theta_G = []
phi_G = []

for i in range(num_el):
    position.append(pos0)
    if (i% 100 == 0):
        print '-------- starting electron:', i
    e_i = electron(E0, pos0, dir0)

    backscattered = False
    while ((not backscattered) and (e_i.energy>=Emin)):# not backscattered
    #for i in range(2):
        #print ' electron at position', e_i.xyz

        # new instance of scatter
        scatter_i = scatter(e_i, material, Wc, tables_moller, tables_gryz)

        # let the electron travel depending on the model used
        scatter_i.compute_pathl()
        #print 'Path length is:', scatter_i.pathl

        # update electron position
        e_i.update_xyz(scatter_i.pathl)
        #print 'new position is', e_i.xyz
        position.append(e_i.xyz)

        if (e_i.xyz[2]<= 0.):
            backscattered = True
            #print 'backscattered', e_i.xyz[2]
            BSE.append(float(e_i.energy))
            count += 1


        # determine scattering type
        scatter_i.det_type()
        #print 'Scatter type is:', scatter_i.type

        #print 'electron energy is:', e_i.energy
        # determine energy loss
        scatter_i.compute_Eloss()
        #print 'Energy loss is:', scatter_i.E_loss

        # update electron energy
        e_i.update_energy(scatter_i.E_loss)
        #print 'new energy is:', e_i.energy

        # determine scattering angles
        scatter_i.compute_sAngles()
        #print ' half phi is:', scatter_i.halfPhi
        #print 'cos square half theta is:', scatter_i.c2_halfTheta
        if (scatter_i.type == 'Rutherford'):
            phi_R.append(2.*scatter_i.halfPhi)
            theta_R.append(2.*acos(scatter_i.c2_halfTheta**0.5))
        elif('Gryzinski' in scatter_i.type):
            phi_G.append(2.*scatter_i.halfPhi)
            theta_G.append(2.*acos(scatter_i.c2_halfTheta**0.5))

        # update electron new traveling direction
        e_i.update_direction(scatter_i.c2_halfTheta, scatter_i.halfPhi)
        #print
print 'total BSE electrons', count

fileBSE = 'BSE_50tilt.out'
with open(fileBSE, 'w') as f:
    for item in BSE:
        f.write("%s\n" % item)

print 'BSE data was written to', fileBSE


file_pos = 'positions_50tilt.out'
with open(file_pos, 'w') as f:
        for item in position:
            f.write("%s\n" % item)

print 'pos data was written to', file_pos



file_phiR = 'phiR_50tilt.out'
with open(file_phiR, 'w') as f:
        for item in phi_R:
            f.write("%s\n" % item)

print 'phi angles for Rutherford was written to', file_phiR


file_thetaR = 'thetaR_50tilt.out'
with open(file_thetaR, 'w') as f:
        for item in theta_R:
            f.write("%s\n" % item)

print 'theta angles for Rutherford was written to', file_thetaR


file_phiG = 'phiG_50tilt.out'
with open(file_phiG, 'w') as f:
        for item in phi_G:
            f.write("%s\n" % item)

print 'phi angles for Gryzinski was written to', file_phiG

file_thetaG = 'thetaG_50tilt.out'
with open(file_thetaG, 'w') as f:
        for item in theta_G:
            f.write("%s\n" % item)

print 'theta angles for Gryzinski was written to', file_thetaG
