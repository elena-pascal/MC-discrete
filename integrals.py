# A number of numerical integration procedures
# for generating the discrete cross section integral tables
import numpy as np

def extF_limits_moller(E, Ec):
    '''
	returns the limits of intergration for the moller excitation function
	in  :: E, Ec
	out :: a, b
	'''
    a = Ec/E
    b = 0.5
    return (a, b)

def extF_limits_gryz(E, Ei):
    '''
	return the limits of intergration of the gryzinski excitation function
	in  :: E, Ei
	out :: a, b
	'''
    a = Ei/E
    b = (1. + Ei/E)/2.
    return (a, b)


def trapez(a, b, E, n_e, ext_func, nSect, *Wc):
    '''
    the usual numerical trapezoidal integration
    a, b are the limits of integration
    f is a function of energy
    nSect is the number of sections
    intT is the cummulative integral at point n
    '''

    #the size of a step is determined by the number of chosen sections
    dx = (b - a)/nSect

    #initialise the sum of f(x) for inner values (1..n-1) of x
    sum_inner = 0.

    for indx in np.arange(1, nSect-1):
        W = a + indx*dx
        sum_inner +=  ext_func(E, W, n_e, *Wc)

    intT = ( ext_func(E, a, n_e, *Wc) + ext_func(E, b, n_e, *Wc) )*dx/2. + dx*sum_inner
    return intT
