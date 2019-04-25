import numpy as np

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
Emin = UnitScalar(10000, units = 'eV') # eV
tilt = 60 # degrees
pos0 = np.array([0., 0., 0.,])
dir0 = np.array([-np.sin(np.radians(tilt)), 0. , np.cos(np.radians(tilt))])
model = 'DS' # discrete scattering

nBinsW = 10
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

for i in range(num_el):
    print '-------- starting electron:', i
    e_i = electron(E0, pos0, dir0)

    while ((e_i.xyz[2]>=0.) and (e_i.energy>=Emin)):# not backscattered
        print ' electron at position', e_i.xyz

        # new instance of scatter
        scatter_i = scatter(e_i, material, Wc, tables_moller, tables_gryz)

        # let the electron travel depending on the model used
        scatter_i.compute_pathl()
        print 'Path length is:', scatter_i.pathl

        # update electron position
        e_i.update_xyz(scatter_i.pathl)
        print 'new position is', e_i.xyz

        # determine scattering type
        scatter_i.det_type()
        print 'Scatter type is:', scatter_i.type

        print 'electron energy is:', e_i.energy
        # determine energy loss
        scatter_i.compute_Eloss()
        print 'Energy loss is:', scatter_i.E_loss

        # update electron energy
        e_i.update_energy(scatter_i.E_loss)
        print 'new energy is:', e_i.energy

        # determine scattering angles
        scatter_i.compute_sAngles()
        print 'cos square half phi is:', scatter_i.c2_halfPhi
        print 'half theta is:', scatter_i.halfTheta

        # update electron new traveling direction
        e_i.update_direction(scatter_i.c2_halfPhi, scatter_i.halfTheta)

        print
