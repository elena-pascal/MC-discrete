import numpy as np
from math import acos
import time
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

num_el = 1000
E0 = UnitScalar(3000, units = 'eV') # eV
Emin = UnitScalar(100, units = 'eV') # eV
tilt = 45. # degrees
pos0 = np.array([0., 0., 0.,])
dir0 = np.array([0., -np.sin(np.radians(tilt)) , np.cos(np.radians(tilt))])
model = 'DS' # discrete scattering

nBinsW = 300
nBinsE = 200

Wc = UnitScalar(10, units = 'eV')

def u2n(value_with_units):
    '''
    Tranforms quantity with units in
    numpy unitless array
    '''
    return np.array(value_with_units)


print '---- calculating Moller tables'
start1_time = time.time()
# remove the function dependece on the constant and get rid of units
funcToint_M = lambda E, W, n_e : u2n(moller_dCS(E, W, n_e, c_pi_efour))

tables_moller = trapez_table( float(E0), float(Emin), np.array([Wc]), float(material.fermi_e()), \
            np.array([material.nval()]), funcToint_M, nBinsW, nBinsE)
#print tables_moller
print("--- %s seconds ---" % (time.time() - start1_time))
print

e_i = electron(E0, pos0, dir0)

start2_time = time.time()
print '---- calculating Gryzinski tables'
tables_gryz = []

funcToint_G = lambda E, W, n_e, Ebi : u2n(gryz_dCS(E, W, n_e,\
                                            c_pi_efour, Ebi))

tables_gryz=trapez_table(float(E0), float(Emin), np.array(material.Es()), float(material.fermi_e()),\
                np.array(material.ns()), funcToint_G, nBinsW, nBinsE )
#print tables_gryz
print("--- %s seconds ---" % (time.time() - start2_time))
print

# new instance of scatter
scatter_i = scatter(e_i, material, Wc, tables_moller, tables_gryz)

scatter_i.det_type()



print '---- compute pathl 1000 times'
start3_time = time.time()
# let the electron travel depending on the model used
for i in range(1000):
    scatter_i.compute_pathl()
print("--- %s seconds ---" % (time.time() - start3_time))
print


print '---- update position 1000 times'
start4_time = time.time()
# update electron position
for i in range(1000):
    e_i.update_xyz(scatter_i.pathl)
print("--- %s seconds ---" % (time.time() - start4_time))
print

print '---- determine type 1000 times'
start5_time = time.time()
# determine scattering type
for i in range(1000):
    scatter_i.det_type()
print("--- %s seconds ---" % (time.time() - start5_time))
print

scatter_i.E_loss = 200
scatter_i.type = 'Gryzinski1s'

print '---- compute angles 1000 times'
start6_time = time.time()
for i in range(1000):
    scatter_i.compute_sAngles()
print("--- %s seconds ---" % (time.time() - start6_time))
print

print '---- update directions 1000 times'
start7_time = time.time()
for i in range(1000):
    e_i.update_direction(scatter_i.c2_halfTheta, scatter_i.halfPhi)
print("--- %s seconds ---" % (time.time() - start7_time))
print


print '---- compute energy losses 1000 times'
start8_time = time.time()
for i in range(1000):
    scatter_i.compute_Eloss()
print("--- %s seconds ---" % (time.time() - start8_time))
print

# update electron energy
print '---- update energy 1000 times'
start9_time = time.time()
for i in range(1000):
    e_i.update_energy(scatter_i.E_loss)
print("--- %s seconds ---" % (time.time() - start9_time))
print
