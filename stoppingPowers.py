from numpy import log

from scimath.units.api import has_units
from scimath.units.energy import  eV
from scimath.units.length import  angstrom, cm, m
from scimath.units.dimensionless import dim

from parameters import pi_efour, bohr_r


###################################################################
#                       Total stopping power                      #
###################################################################

# 2a) Moller stopping power for free electrons
@has_units
def moller_sp(E, Emin, nfree, n, c_pi_efour=pi_efour):
    """ Calculate the Moller stopping power per unit length

    Parameters
    ----------
    E      : array : units = eV

    Emin   : array : units = eV

    nfree  : array : units = dim

    n      : array : units = m**-3
           atom number density

    c_pi_efour: scalar: units = cm**2 * eV**2

    Returns
    -------
    sp_M   : array : units = eV/angstrom
    """
    eps = Emin*1./E
    sp_M = nfree*n*c_pi_efour*(2. - (1./(1.-eps)) +  log( 1./(8.* eps*((1.-eps)**2)) ))/E \
                            * m**-3 * cm**2/angstrom**-1
    return sp_M


# 2b) Gryzinski stopping power for core shell electrons
@has_units
def gryz_sp(E, Enl, nsi, n, c_pi_efour=pi_efour):
    """ Calculate the Gryzinski inelastic cross section per unit length

        Parameters
        ----------
        E      : array : units = eV

        Enl    : array : units = eV

        nsi    : array : units = dim
               number of electrons in i-shell

        n      : array : units = m**-3
               atom number density

        c_pi_efour: scalar: units = cm**2 * eV**2

        Returns
        -------
        s_G    : array : units = eV/angstrom
    """

    U = E*1./Enl
    sp_G = nsi * n* c_pi_efour * ((U - 1.)/(U + 1.))**1.5 * (log(U) +
                                     4.*log(2.7+(U-1.)**0.5)/3.)/E \
                                     * m**-3 * cm**2/angstrom**-1
    return sp_G

# 2c) Quinn stopping power for plasmons
@has_units
def quinn_sp(E, Epl, Ef, n, c_bohr_r=bohr_r):
    """ Calculate the Quinn inelastic stopping power per unit length
        Patrick's formula

        Parameters
        ----------
        E      : array : units = eV
                incident energy

        Epl    : array : units = eV
                plasmon energy

        Ef     : array : units = eV
                Fermi energy

        n      : array : units = m**-3
               atom number density

        bohr_r :scalar : units = m

        Returns
        -------
        sp_Q    : array : units = eV/angstrom
    """
    E_Ef = E*1./Ef
    Epl_Ef = Epl*1./Ef

    s_Q_total = Epl * log( ((1. + Epl_Ef)**(0.5) - 1.)/ ( E_Ef**0.5 -
             (E_Ef - Epl_Ef)**0.5 ) )/(2. *  bohr_r * E)
    sp_Q = s_Q_total * m**(-1)/ angstrom**(-1) * Epl

    return sp_Q

############################## Bethe #######################################

# 2d) Bethe continuous stopping power
@has_units
def bethe_cl_sp(Z, E, n, c_pi_efour=pi_efour):
    """ Calculate the Bethe continuous inelastic scattering stopping power
        per unit length

        Parameters
        ----------
        Z      : array : units = dim
                atomic number

        E      : array : units = eV
                incident energy

        n      : array : units = m**-3
               number density

        c_pi_efour: scalar: units = cm**2 * eV**2

        Returns
        -------
        sp_B    : array : units = eV/angstrom
    """
    #the mean ionisation potential J in eV
    if (Z>=13):
        J = (9.76*Z + 58.5/(Z**0.19))
    else:
        J = 11.5*Z

    sp_B = 2.*c_pi_efour*n*(Z/E) * log( 1.166*(E )/J ) * m**-3*cm**2/angstrom**-1
    return sp_B



@has_units
def bethe_mod_sp_k(Z, E, n, k, c_pi_efour=pi_efour):
    """ Calculate the Bethe continuous inelastic scattering stopping power
        per unit length using Joy and Luo potential (1989)

        Parameters
        ----------
        Z      : array : units = dim
                atomic number

        E      : array : units = eV
                incident energy

        n      : array : units = m**-3
               number density

        c_pi_efour: scalar: units = cm**2 * eV**2

        Returns
        -------
        sp_B    : array : units = eV/angstrom
    """
    #the mean ionisation potential J in eV
    if (Z>=13):
        J = (9.76*Z + 58.5/(Z**0.19))
    else:
        J = 11.5*Z

    sp_B = 2.*c_pi_efour*n*(Z/E) * log( 1.166*(E + k*J )/J ) * m**-3*cm**2/angstrom**-1
    return sp_B


@has_units
def bethe_mod_sp(E, n, Zi, Ei, Zval, Eval, c_pi_efour=pi_efour):
    """ Calculate the Bethe continuous inelastic scattering stopping power
        per unit length using explicitely the binding energy for all the shells

        Parameters
        ----------
        E      : array : units = eV
                incident energy

        n      : array : units = m**-3
               number density

        Zi     : array : units = dim
                occupancy of shell i

        Ei     : array : units = eV
                binding energy of shell i

        Zval   : array : units = dim
               occupancy of valence shell

        Eval   : array : units = eV
               binding energy of valence electrons

        c_pi_efour: scalar: units = cm**2 * eV**2

        Returns
        -------
        sp_B    : array : units = eV/angstrom
    """

    prefactor = (2.*c_pi_efour*n/E ) * m**-3*cm**2/angstrom**-1

    sumi = Zval * log(E*1./Eval)
    for indx, Eb in enumerate((Ei)):
        sumi = sumi + Zi[indx] * log( E*1./Eb )

    sp_B = prefactor * sumi
    return sp_B
