from scimath.units.energy import J, eV, KeV
from scimath.units.api import UnitScalar, UnitArray, convert, has_units

from scipy.constants import pi, Avogadro, hbar, m_e, e, epsilon_0, eV

def exp_params():
    params = {'file': 'parameters.py'}

    # incident energy
    params['E'] = UnitScalar(20000, units="eV")

    # cut off energy value for free electron scattering energy transfer [5..100]
    params['Ec'] = UnitScalar(10., units="eV")

    return params



class electron:
    ''' An electron is an object with defined energy position and
        direction in the beam frameself.
        Input can have units.'''
    def __init__(self, energy, position, direction):
        self.energy = energy
        self.xyz = position
        self.dir = direction
