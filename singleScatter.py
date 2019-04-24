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

nBinsW = 10
nBinsE = 10

Wc = 100

def u2n(value_with_units):
    '''
    Tranforms quantity with units in
    numpy unitless array
    '''
    return np.array(value_with_units)

# remove the function dependece on the constant and get rid of units
funcToint_M = lambda E, W, n_e : u2n(moller_dCS(E, W, material.get_nval(), c_pi_efour))

print '---- calculating Moller tables'
a_M, b_M = u2n(extF_limits_moller(E0, Wc))
tables_moller = trapez_table(a_M, b_M, Emin, E0, material.get_nval(), funcToint_M, nBinsW, nBinsE)
# print tables_moller

print '---- calculating Gryzinski tables'
a_G, b_G = u2n(extF_limits_gryz(E0, material.get_Es()) )
tables_gryz = []
for ishell in range(len(material.get_ns())):
    funcToint_G = lambda E, W, n_e, Ebi : u2n(gryz_dCS(E, E, material.get_ns()[ishell], c_pi_efour, Ebi[ishell]))
    tables_gryz.append(trapez_table(a_G, b_G, Emin, E0, material.get_ns()[ishell], funcToint_M, nBinsW, nBinsE) )

for i in range(num_el):
    print '-------- starting electron:', i
    e_i = electron(E0, pos0, dir0)

    while ((e_i.xyz[2]>=0.) and (e_i.energy>=Emin)):# not backscattered
        print ' electron at position', e_i.xyz

        # new instance of scatter
        scatter_i = scatter(e_i, material, Wc, tables_moller, tables_gryz)

        # let the electron travel depending on the model used
        scatter_i.compute_patl()
        print 'Path length is:', scatter_i.pathl

        # update electron position
        electron_i.update_xyz(scatter_i.pathl)

        # determine scattering type
        scatter_i.det_type()
        print 'Scatter type is:', scatter.type

        # determine energy loss
        scatter.compute_Eloss()
        print 'Energy loss is:', scatter_i.E_loss

        # update electron energy
        e_i.update_energy(scatter_i.E_loss)

        # determine scattering angles
        scatter_i.calculate_sAngles()
        print 'cos square half phi is:', scatter_i.c2_halfPhi
        print 'half theta is:', scatter_i.halfTheta

        # update electron new traveling direction
        e_i.update_dir(scatter.c2_halfPhi, scatter.halfTheta)
