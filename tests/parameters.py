from scimath.units.api import UnitScalar
import numpy as np
from scipy.constants import hbar, m_e, e, epsilon_0, physical_constants


# constants without units
pi_efour = 6.51408491409531e-14
bohr_r = physical_constants['Bohr radius'][0]

# constants with units
u_hbar       = UnitScalar(hbar, units="J*s")
u_me         = UnitScalar(m_e, units="kg")
u_e          = UnitScalar(e, units="coulomb")
u_eps0       = UnitScalar(epsilon_0, units="farad*m**-1")
u_pi_efour = UnitScalar(6.51408491409531e-14, units="cm**2 * eV**2")
u_bohr_r     = UnitScalar(physical_constants['Bohr radius'][0], units="m")





# detector Parameters from Patrick's paper
L = 15250 # microns
sigma = 70 # degrees
theta_C = 0
alpha = 90 - sigma + theta_C # degrees
delta = 49.375 # microns
x_pixels = 640
y_pixels = 480
x_PC = 3.57
y_PC = 113.45

xy_PC = np.array([x_PC*delta, y_PC*delta])
