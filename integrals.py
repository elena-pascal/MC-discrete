# A number of numerical integration procedures
# for generating the discrete cross section integral tables
import numpy as np

# define some Python errors for the m values
class Error(Exception):
    ''' Bare class for exceptions'''
    pass

class mTooSmall(Error):
    ''' Raised when the iteration step m is negative'''
    pass

class mTooLarge(Error):
    ''' Raised when the iteration step m becomes too large'''
    pass



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

def trapez_refine(a, b, E, n_e, ext_func, m, *Wc):
    '''
    m step of refinement for the recursive trapezoidal integration
    ext_func is the function to be integrated between the limits a and b
    when called with m=0 the function returns the trapezoidal integration with one bin
    subsequent calls with higher m increase the number of bins by 2**m
    and return the new terms in this step
    '''
    try:
        if (m < 0):
            raise mTooSmall
            sys.exit()
    except mTooSmall:
        print ' Fatal error! in trapez_refine'
        print ' Illegal input value of m.'
        print ' Stopping.'

    if (m == 0):
        # usual trapezoidal integration
        newTerms = 0.5*(b - a) * ( ext_func(E, a, n_e, *Wc) + ext_func(E, b, n_e, *Wc) )

    elif ( m >= 1 ):
    	# step size is
        h = (b-a)/ 2**m
        sum_m = 0.
        for i in np.arange(1, 2**(m-1)):
            x = a + (2*i - 1)*h
            sum_m = sum_m + ext_func(E, x, n_e, *Wc)

	    #intm = trapez_refine(a, b, E, n_e, ext_func, m-1, Wc) * 0.5 + h * sum_m
        newTerms = h * sum_m

    return newTerms



def trapez_tol(a, b, E, n_e, ext_func, tol, *Wc):
    '''
    call trapez_refine with increasing m until tolerance is reached
    m is the step number, will need this to determine the number of
    equidistant points at which the intergral was evaluated
    returns an array with the integral under the curve
    evaluated at all the steps m
    '''

    # set the max value of m steps before we decide they are too many
    maxm = 20

    tolReached = False
    fatalError = False

    # array to contain the integral at step m
    totalInt = np.empty(maxm)

    totalInt[0] = trapez_refine(a, b, E, n_e, ext_func, 0, *Wc)

    m = 1
    while ((not tolReached) and (not fatalError)):
        newTerms = trapez_refine(a, b, E, n_e, ext_func, m, *Wc)
        totalInt[m] = newTerms + 0.5 * totalInt[m-1]
        print 'm:', m
        rel_diff = abs((totalInt[m]-totalInt[m-1])/totalInt[m])
        print 'diff', rel_diff
        if (rel_diff < tol):
            tolReached = True

        try:
            m += 1
            if (m >= maxm):
                raise mTooLarge
        except mTooLarge:
            fatalError = True
            print
            print 'Failed to converge in 20 steps in trapez_tol'
            print
        


    return totalInt[m-1]
