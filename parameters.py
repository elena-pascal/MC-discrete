from scimath.units.energy import J, eV, KeV
from scimath.units.api import UnitScalar, UnitArray, convert, has_units

from scipy.constants import pi, Avogadro, hbar, m_e, e, epsilon_0, eV

u_hbar = UnitScalar(hbar, units="J*s")
u_me = UnitScalar(m_e, units="kg")
u_e    = UnitScalar(e, units="coulomb")
u_eps0 = UnitScalar(epsilon_0, units="farad*m**-1")
c_pi_efour = UnitScalar(6.51408491409531e-14, units="cm**2 * eV**2")



def u2n(value_with_units):
    '''
    Tranforms quantity with units in
    numpy unitless array
    '''
    return np.array(value_with_units)
