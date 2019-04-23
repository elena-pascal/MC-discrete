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
    a = np.array([Ec])
    b = np.array([0.5 * E])
    return (a, b)

def extF_limits_gryz(E, Ei):
    '''
	return the limits of intergration of the gryzinski excitation function
	in  :: E, Ei
	out :: a, b
	'''
    a = Ei
    b = (E + Ei) * 0.5
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

    for indx in np.arange(1, nSect): # [1, nSect) = [1, nSect-1]
        W = a + indx*dx
        sum_inner +=  ext_func(E, W, n_e, *Wc)

    intT = ( ext_func(E, a, n_e, *Wc) + ext_func(E, b, n_e, *Wc) )*dx/2. + dx*sum_inner
    return intT


def trapez_table(Wmin, Wmax, Emin, Emax, n_e, ext_func, nBinsW, nBinsE):
    '''
    As above but return a table of integrals for different energy losses and incident energies
    int_0^Wi for all incident energies Ei and all energy losses Wi
    The way the binning is considered is that the value of the bin is taken to be upper
    bound of the bin
    '''
    tables = np.zeros(len(n_e))

    for shell in range(len(n_e)):
        int_extFunc = np.empty([nBinsE, nBinsW]) # [0:nBinsE-1], [0:nBinsW-1]


        # the size of a step in energy loss W is determined by the number of chosen sections nBinsW
        dW = (Wmax[shell] - Wmin[shell])/nBinsW

        # the size of a step in incident energy E is determined by the number of chosen sections nBinsE
        dE = (Emax - Emin)/nBinsE

        # initialise the sum of f(x) for inner values (1..n-1) of x
        sum_innerW = np.zeros(nBinsW)

        for indx_E in np.arange(nBinsE): # [1, nSectE]
            Ei = Emin + indx_E*dE

            for indx_W in np.arange(nBinsW-1):
                Wi = Wmin[shell] + indx_W*dW
                sum_innerW[indx_W] = sum_innerW[indx_W-1] + ext_func(Ei, Wi, n_e[shell], *Wmin[shell]) # sum_inner[0] = 0

                int_extFunc[indx_E, indx_W] = ( ext_func(Ei, Wmin[shell], n_e[shell], *Wmin[shell]) + ext_func(Ei, Wi, n_e[shell]), *Wmin[shell] )*dW/2. \
                                                    + dW * sum_innerW[indx_W-1]
                # last value and total area integral
                int_extFunc[indx_E, nBinsW-1] = ( ext_func(Ei, Wmin[shell], n_e[shell], *Wmin[shell]) + ext_func(Ei, Wmax[shell], n_e[shell]), *Wmin[shell] )*dW/2. \
                                                + dW * sum_innerW[nBinsW-2]

                x = np.linspace(Wmin[shell], Wmax[shell], nBinsW)
                y = np.linspace(Emin, Emax, nBinsE)
                xx, yy = np.meshgrid(x, y)

                tables[shell] = [xx, yy, int_extFunc[1:nBinsE, 1:nBinsW]]
    return tables


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

        rel_diff = abs((totalInt[m]-totalInt[m-1])/totalInt[m])
        if (rel_diff < tol):
            tolReached = True
            return totalInt[m]

        try:
            m += 1
            if (m >= maxm):
                raise mTooLarge
        except mTooLarge:
            fatalError = True
            print
            print 'Failed to converge in 20 steps in trapez_tol'
            print



def int_Romberg(a, b, E, n_e, ext_func, *Wc):
    '''
    Romberg procedure uses extrapolation to calculate numerical
    estimates of the definite integral R = int_a^b{f(x)dx}.
    The input is the function f, the endpoints of the interval
    [a,b], the acceptable error untill which iterations are added.
    '''
    tolReached = False
    fatalError = False

    maxm = 20

    tableR = np.zeros((maxm, maxm), dtype=np.float64)
    pow_4 = 4 ** np.arange(maxm, dtype=np.float64) - 1

    # trapezoidal integral for R[0,0]
    h = b - a
    tableR[0, 0] = h * (ext_func(E, a, n_e, *Wc) + ext_func(E, b, n_e, *Wc)) / 2

    for m in np.arange(1, maxm):
        h = h/2.

        # extended trapezoidal rule
        tableR[m, 0] = tableR[m - 1, 0] / 2
        tableR[m, 0] += h * np.sum(ext_func(E, a + i*h, n_e, *Wc) \
                                    for i in range(1, 2**m + 1, 2))

        # richardson extrapolation
        for k in np.arange(1, m + 1):
            tableR[m, k] = tableR[m, k - 1] + \
                  (tableR[m, k - 1] - tableR[m - 1, k - 1]) / pow_4[k]


    return tableR[-1, -1]


def logSpace(a, b, E, n_e, ext_func, nSeg, *Wc):
    '''
    if the function looks linear in log-log space
    we could do the intergral there instead
    as this should require fewer steps
    '''
    # the size of a step is determined by the number of chosen sections
    x = np.logspace(a, b, num=nSeg+1)
    print a, b, x
    # the left edge of first segment in log space
    logx = np.log(x)

    # the value of the function at this point:
    logy = np.log(ext_func(E, logx, n_e, *Wc))

    intT = 0.

    # calculate the slopes mi of all the segments in log-log space
    for i in np.arange(1, nSeg):

        # calculate the slope of this segment in log space
        m = (logy[i]- logy[i+1])/ (logx[i]- logx[i+1])

        mp1 = m + 1

        # calculate the y-intercept in log space
        n = logy[i] - (m*logx[i])

        # now compute the integral under the segment in linear space
        if(m is -1):# put a tolerance here
            intSeg = np.exp(n) * np.log(x[i+1]/x[i])
        else:
            intSeg = np.exp(n) * (x[i+1]**mp1 - x[i]**mp1) / mp1
            #intSeg = np.exp(n) * (x[i+1]**m - x[i]**m)/m

        intT = intT + intSeg

    return intT
