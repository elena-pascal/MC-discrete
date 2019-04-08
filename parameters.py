from scimath.units.energy import J, eV, KeV
from scimath.units.api import UnitScalar, UnitArray, convert, has_units



def exp_params():
    params = {'file': 'parameters.py'}

    # incident energy
    params['E'] = UnitScalar(20000, units="eV")

    # cut off energy value for free electron scattering energy transfer [5..100]
    params['Ec'] = UnitScalar(10., units="eV")

    return params
