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

def trapez_refine(a, b, E, n_e, ext_func, m, Wc):
    '''
    m step of refinement for the recursive trapezoidal integration
    ext_func is the function to be integrated between the limits a and b
    when called with m=0 the function returns the trapezoidal integration with one bin
    subsequent calls with higher n increase the number of bins by 2**m
    '''
    if (m < 0):
        write ( *, '(a)' ) ' ------------------------------'
        write ( *, '(a)' ) ' Fatal error! in trapez_refine'
        write ( *, '(a)' ) ' Illegal input value of m.'
        write ( *, '(a)' ) ' Stopping.'
        write ( *, '(a)' ) ' ------------------------------'
        return none

    else if (m == 0):
    # usual trapezoidal integration
        intm = 0.5*(b - a) * ( ext_func(E, a, n_e, Wc) + ext_func(E, b, n_e, Wc) )

    else if ( m >= 1 ):
	# step size is
	h = (b-a)/ 2**m

	sum_m = 0.
	for i in np.arange(1, 2**(m-1)):
	   x = a + (2*i - 1)*h
	   sum_m = sum_m + ext_func(E, x, n_e, Wc)

	intm = trapez_refine(a, b, E, n_e, ext_func, m-1, Wc) * 0.5 + h * sum_m
    return intm



def trapez_tol(a, b, E, n_e, ext_func, tol, intT, Wc):
    '''
    call trapez_refine with increasing m until tolerance is reached
    m is the step number, will need this to determine the number of
    equidistant points at which the intergral was evaluated
    returns an array with the integral under the curve
    evaluated at all the steps m
    '''

    m = 1
    tolReached = FALSE
    fatalError = FALSE

    do while ((.NOT.tolReached).AND.(.NOT.fatalError))

    end do
     
