import numpy as np

from scattering import moller_dCS
from integrals import trapez, extF_limits_moller
from material import scattering_params
from parameters import exp_params

from scimath.units.energy import J, eV, KeV
from scimath.units.electromagnetism import coulomb, farad
from scimath.units.length import m, cm, km, angstrom
from scimath.units.api import UnitScalar, UnitArray, convert, has_units
from scipy.constants import pi, Avogadro, hbar, m_e, e, epsilon_0, eV

u_hbar = UnitScalar(hbar, units="J*s")
u_me = UnitScalar(m_e, units="kg")
u_e    = UnitScalar(e, units="coulomb")
u_eps0 = UnitScalar(epsilon_0, units="farad*m**-1")
c_pi_efour = UnitScalar(6.51408491409531e-14, units="cm**2 * eV**2")

def u2n(value_with_units):
    return np.array(value_with_units)



species = 'Al'
sp_Al = scattering_params('Al')

experiment = exp_params()

a, b = u2n(extF_limits_moller(experiment['E'], experiment['Ec']))


def funcToint(x, y, z):
    return np.array(moller_dCS(x, y, z, c_pi_efour))


print 'integral', trapez(a, b, u2n(experiment['E']), u2n(sp_Al['n_val']), funcToint, 100000)
