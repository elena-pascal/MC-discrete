import numpy as np

from scattering import moller_dCS
from integrals import trapez, trapez_table, trapez_tol, int_Romberg, logSpace, extF_limits_moller
from material import scattering_params
from parameters import exp_params
from parameters import u_hbar, u_me, u_e, u_eps0, c_pi_efour

from scimath.units.energy import J, eV, KeV
from scimath.units.electromagnetism import coulomb, farad
from scimath.units.length import m, cm, km, angstrom
from scimath.units.api import UnitScalar, UnitArray, convert, has_units


def u2n(value_with_units):
    '''
    Tranforms quantity with units in
    numpy unitless array
    '''
    return np.array(value_with_units)



species = 'Al'
sp_Al = scattering_params('Al')

experiment = exp_params()

a, b = u2n(extF_limits_moller(experiment['E'], experiment['Ec']))


def funcToint(const):
    ''' remove units and dependence on constant '''
    return lambda x, y, z : u2n(moller_dCS(x, y, z, const))

ext_func = funcToint(c_pi_efour)

print 'integral with trapez', trapez(a, b, u2n(experiment['E']), u2n(sp_Al['n_val']), ext_func, 10000)
print

print 'table integral', trapez_table(a, b, u2n(experiment['Ec']), u2n(experiment['E']), u2n(sp_Al['n_val']), ext_func, 100, 1000)
print


#print 'integral with refined trapez', trapez_tol(a, b, u2n(experiment['E']), u2n(sp_Al['n_val']), ext_func, 0.001)
#print
#print 'integral with Romberg', int_Romberg(a, b, u2n(experiment['E']), u2n(sp_Al['n_val']), ext_func)
#print

#print 'integral in log-log space',  logSpace(1, 5, u2n(experiment['E']), u2n(sp_Al['n_val']),  ext_func, 100000)