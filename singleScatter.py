import numpy as np
from electron import electron
from material import material
from scattering import scatter
from extFunctions import gryz_dCS, moller_dCS
from integrals import trapez_table, extF_limits_gryz, extF_limits_moller
from parameters import c_pi_efour

material = material('Al')

num_el = 100
E0 = 20000 # eV
Emin = 10000 # eV
tilt = 60 # degrees
pos0 = [0., 0., 0.,]
dir0 = [-np.sin(np.radians(tilt)), 0. , np.cos(np.radians(tilt))]
model = 'DS' # discrete scattering

nBinsW = 100
nBinsE = 1000

Wc = 10

def u2n(value_with_units):
    '''
    Tranforms quantity with units in
    numpy unitless array
    '''
    return np.array(value_with_units)

def funcToint_M(const):
    ''' remove units and dependence on constant '''
    return lambda x, y, z : u2n(moller_dCS(x, y, z, const))

def funcToint_G(const):
    ''' remove units and dependence on constant '''
    return lambda x, y, z, w : u2n(gryz_dCS(x, y, z, const, w))

a_M, b_M = u2n(extF_limits_moller(E0, Wc))
a_G, b_G = u2n(extF_limits_gryz(E0, material.get_Es()))

ext_func_M = funcToint_M(c_pi_efour)
ext_func_G = funcToint_G(c_pi_efour)

tables_moller = trapez_table(a_M, b_M, Emin, E0, material.get_ns(), ext_func_M, nBinsW, nBinsE)
tables_gryz = trapez_table(a_G, b_G, Emin, E0, material.get_nval(), ext_func_G, nBinsW, nBinsE)

for i in num_el:
    ei = electron(E, pos0, dir0)
    while ((electron.xyz[2]>=0.) and (electron.energy>=Emin)):# not backscattered
        # new instance of scatter
        scatter = scatter(electron, material, Ec, tables_moller, tables_gryz)

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
