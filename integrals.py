

def trapez(a, b, E, n_e, ext_func, nSect, intT, Wc):
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

    for indx in np.linear(1, nSect-1):
        W = a + indx*dx
        sum_inner+ =  ext_func(E, W, n_e, Wc)

    intT = ( ext_func(E, a, n_e, Wc) + ext_func(E, b, n_e, Wc) )*dx/2. + dx*sum_inner
