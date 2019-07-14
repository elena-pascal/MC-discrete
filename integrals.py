# A number of numerical integration procedures
# for generating the discrete cross section integral tables
import numpy as np
from errors import mTooLarge, mTooSmall

def binMiddles(Xmin, Xmax, numBins):
    ''' how many times do I have to see this written wrong?
        For a given range and number of bins
        return the numpy array of the bins middles
    '''
    dX = (Xmax-Xmin)/numBins
    return np.linspace(Xmin+dX/2, Xmax-dX/2, num=numBins, retstep=True)

def binEdges(Xmin, Xmax, numBins):
    ''' how many times do I have to see this written wrong?
        For a given range and number of bins
        return the numpy array of the bins edges and the step size
    '''
    return np.linspace(Xmin, Xmax, num=numBins+1, retstep=True)



def extF_limits_moller(E, Ec, Ef):
    '''
	returns the limits of intergration for the moller excitation function
	in:
        E :: maximum energy to consider
        Ec ::  minimum energy to consider
	out:
        a :: lower integral limit
        b :: upper integral limit
	'''

    # a = Ec

    b =  (E - Ef)*0.5
    return b

def extF_limits_gryz(E, Ei, Ef):
    '''
	return the limits of intergration of the gryzinski excitation function
	in  :: E, Ei
	out :: a, b
	'''

    # a = Ei
    #b = (E - Ef + Ei)
    b = E
    return b


def trapez(a, b, E, n_e, ext_func, nSect, *Ebi):
    '''
    the usual numerical trapezoidal integration
    a, b are the limits of integration
    f is a function of energy
    nSect is the number of sections
    intT is the cummulative integral at point n
    gryz ext function has one extra parameter for each shell: the binding energy
    '''

    #the size of a step is determined by the number of chosen sections
    dx = (b - a)/nSect

    #initialise the sum of f(x) for inner values (1..n-1) of x
    sum_inner = 0.

    for indx in np.arange(1, nSect): # [1, nSect) = [1, nSect-1]
        W = a + indx*dx
        sum_inner +=  ext_func(E, W, n_e, *Ebi)

    intT = ( ext_func(E, a, n_e, *Ebi) + ext_func(E, b, n_e, *Ebi) )*dx/2. + dx*sum_inner
    return intT


def trapez_table(Einc, Emin, Wmin, Ef, n_e, shell_name, ext_func, nBinsW, nBinsE):
    '''
    As above but return a table of integrals for different energy losses and incident energies
    int_0^Wi for all incident energies Ei and all energy losses Wi
    The way the binning is considered is that the value of the bin is taken to be upper
    bound of the bin

    Einc = incident electron energy
    Emin = minimum energy for which the electrons are still tracked
    Elossmin = Ec for Moller and Ei[shell] for Gryzinski
    Ef = Fermi energy for this material
    n_e = number of valence electrons for Moller scattering and number of electrons per
            inner shell n_e[shell]
    ext_func = excitation function
    nBinsW = number of W bins
    nBinsE = number of E bins
    '''

    # e contains the array of possible incident energies in the tables
    try:
         e_tables, _ = binEdges(Emin, Einc, nBinsE) # we will bisect left
         if (Emin > Einc):
             raise ERangeError
    except ERangeError as err:
        print ('! Error: the incident energy is larger than the minimum energy')
        print ('E0:', Einc)
        print ('Emin:', Emin)

    # tables is the final multidimensional table dictionary
    tables = {}
    tables['e_tables'] = [e_tables]

    #print ('E tables', e_tables)
    w_tables = np.empty([nBinsE+1, nBinsW+1])


    # the integral fuction has the same shape as w
    int_extFunc = np.empty([nBinsE+1, nBinsW+1]) # [0:n_e-1], [0:nBinsE-1], [0:nBinsW-1]


    for ishell in range(n_e.size):
        # tables is adictionary of the form {'energy':[], shell1:{'w':{'energy_val':[]}, 'integral':[]}, shell2...}
        tables[shell_name[ishell]] = {'w_tables':{}, 'integral':{}}

        # minimum energy that can be lossed by an electron to scatter of this shell
        W_min = Wmin[ishell]

        # simplify the excitation function to depend only on E and W
        if (n_e.size == 1): # Moller
            func = lambda E, W: ext_func(E, W, n_e[ishell])
        else: # Gryzinski
            func = lambda E, W: ext_func(E, W, n_e[ishell], W_min)

        for indx_E, Ei in enumerate(e_tables):
            #print ('Ei at this step:', Ei)

            if (n_e.size == 1): # Moller
                # the upper integral limit depends on Ei
                W_max =  extF_limits_moller(Ei, W_min, Ef)

            else: # Gryzinski
                W_max = extF_limits_gryz(Ei, W_min, Ef)

            # initialise the integral for the recursive function
            int_extFunc[indx_E, 0] = 0.

            w_tables[indx_E, :], dW = binEdges(W_min, W_max, nBinsW)
            #print ('w_tables', w_tables[indx_E, :])

            # actual integral
            funcEdge0 = func(Ei, W_min)
            for indx_W, Wi in enumerate(w_tables[indx_E, 1:]):
                indx_W = indx_W+1
                funcEdge1 = func(Ei, Wi)
                #print ('Wi', Wi)
                #print ('edges', funcEdge0, funcEdge1)
                int_extFunc[indx_E, indx_W] = ( funcEdge0 + funcEdge1  )*dW/2. +\
                                                      int_extFunc[indx_E, indx_W-1]
                #print ('int', int_extFunc[ishell, indx_E, indx_W])
                funcEdge0 = funcEdge1
            #print ('integral list', int_extFunc[ishell, indx_E, :]/int_extFunc[ishell, indx_E, -1])

        # append [ishell, [w_tables], [int_extFunc]] for every shell
            tables[shell_name[ishell]]['w_table'][str(Ei)] = w_tables
            tables[shell_name[ishell]]['integral'][str(Ei)] = int_extFunc



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
    except mTooSmall as err:
        print (' Fatal error! in trapez_refine:', err)
        print (' Illegal input value of m.')
        print (' Stopping.')
        sys.exit()

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
        except mTooLarge as err:
            fatalError = True
            print( '! Error:', err)
            print ('Failed to converge in 20 steps in trapez_tol')
            print()



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
    print (a, b, x)
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
