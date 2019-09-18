import numpy as np
from scipy.integrate import quad



from extFunctions import gryz_dCS, moller_dCS
from material import material



def maxW_moller(E, Ef):
    '''
	returns the lower limits of intergration for the moller excitation function
	in:
        E  :: maximum energy to consider
        Ef :: Fermi energy
	'''

    return (E - Ef)*0.5

def maxW_gryz(E, Ei, Ef):
    '''
	return the lower limits of intergration of the gryzinski excitation function
	in  :: E, Ef
	'''

    return  E - Ef




def ordersOfMag(maxSize = 100000, numIter = 4):
    ''' iterator over n = [100, 1000, 10000, 100000]'''
    #minSize = 100
    for n in np.geomspace(100, maxSize, num=numIter, dtype=int):
        yield n


def percents(maxSize, numIter = 10):
    '''iterator over n = [n/10, 2*n/10 ... nW]'''
    for n in np.linspace(maxSize/10, maxSize, numIter):
        yield n





class probTable:
    ''' A probability table for Moller and Gryzinski
    energy losses

    P(E, W) = probability of W energy loss for an electron
              of incident energy E

    Format is            E1 E2  ...  Einc
                        __________________
                   W1  | -- --  ...  --
                   W2  | -- --       --
                   .   |
                   .   |
                   .   | NaN
                   Wmax| NaN NaN ...

    The valid values in the table are just the top right triangle.
    '''

    def __init__(self, type, func, E_range, Wmin, tol_E, tol_W, material, target):
        '''
        Define the table

        type  = the type of table, Moller or Gryz + str(shell)
        func  = the excitation function for this type of scattering
        tol_E = relative difference in probabilities between two adiacent Es for same W
        tol_W = relative difference in probabilities between two adiacent Ws for same E

        The two tolerances are not independent of each other.
        '''
        self.type = type
        self.func = func

        self.Emin, self.Emax = E_range[0], E_range[1]
        self.Ef              = material.fermi_e
        self.n_e             = material.params['n_val']
        self.Wmin            = Wmin

        self.tol_E = tol_E
        self.tol_W = tol_W

        self.Es = None
        self.Ws = None

        self.target = target



    def check_intTol(self, Wseries, refVal, integrand=None, W=None):
        ''' For a given Wseries for a set E, check if fine enough'''

        # trapezoidal intergral
        trapezInt = np.trapz(np.array([integrand(W) for W in Wseries]), x=Wseries)

        # relative difference
        diff = abs(trapezInt-refVal)/refVal

        # return the truth value
        return diff < self.tol_W


    def check_probTol(self, Eseries, refVal, integrand=None, W=None):
        ''' For a given Eseries for a set Wseries, check
        if Eseries is fine enough'''

        # energy position for which we test
        E = Eseries[-2]

        print ('E, W:', E, W)
        # simplify the excitation function to depend only on W
        integrand = lambda W: self.func(E, W, self.n_e)

        # trapezoidal intergral in the range [Wmin, W]
        smalldWInt = np.trapz([integrand(self.Wmin), integrand(W)],
                          x = [self.Wmin,               W              ] )

        # gaussian quadrature integral for [Wmin, Wmax]
        totalInt, _ = quad(integrand, self.Wmin, maxW_moller(E, self.Ef), limit=300, epsabs=1e-24)

        # the probability is the fractional integral
        prob = smalldWInt/totalInt
        print ('test prob:', prob)
        print ('ref prob:', refVal)
        # relative difference
        diff = abs(prob - refVal)/refVal

        # return the truth value
        return diff < self.tol_E


    @staticmethod
    def findGeomSeries(startVal, endVal, maxSize, generator, test, refVal, integrand=None, W=None):
        ''' refine the number of values along a given iterator and using
            given acceptance test'''

        withinTol = False
        iterator = generator(maxSize)
        while not withinTol:
            num = next(iterator)

            # choose nW integration points in log space
            Wseries = np.geomspace(startVal, endVal, num=num)

            if test(Wseries, refVal, integrand, W): # within tolerance
                withinTol = True

        return num, Wseries


    @staticmethod
    def findLinSeries(startVal, endVal, maxSize, generator, test, refVal, integrand=None, W=None):
        ''' refine the number of values along a given iterator and using
            given acceptance test'''

        withinTol = False
        iterator = generator(maxSize)
        while not withinTol:
            nE = next(iterator)

            # choose nE integration points in linear space
            Eseries = np.linspace(startVal, endVal, num=nE)

            if test(Eseries, refVal, integrand, W): # within tolerance
                withinTol = True

        return nE, Eseries



    def findWseries(self, E):
        '''
        For the given energy loss tolerance compute the necessary Ws arrays

        We are computing the integrals in the table using the trapezoidal method (np.trapz)
        for speed. For low number of W values this will not be a good approximation since
        the integrand curves are logarithmic.

        Check that the computed integral is within tol_Int of simpy.quad
        '''

        # compute Wmax
        W_max = maxW_moller(E, self.Ef)

        # simplify the excitation function to depend only on E and W
        integrand = lambda W: self.func(E, W, self.n_e)

        # gaussian quadrature integral
        quadInt, error = quad(integrand, self.Wmin, W_max, limit=300, epsabs=1e-24)

        # check if the intergation error is not larger than the tollerance we allow for
        if (error > self.tol_W*quadInt):
            print ('Warning! The W tolarance is smaller than the error of integration')

        # first refine for order of magnitude
        nW1, _ = self.findGeomSeries(startVal = self.Wmin,
                                     endVal = W_max,
                                     maxSize = 100000,
                                     generator = ordersOfMag,
                                     test = self.check_intTol,
                                     refVal = quadInt,
                                     integrand = integrand)

        # second refine in multiple of 10% of the value
        _, series = self.findGeomSeries(startVal = self.Wmin,
                                        endVal = W_max,
                                        maxSize = nW1,
                                        generator = percents,
                                        test = self.check_intTol,
                                        refVal = quadInt,
                                        integrand = integrand)

        # and return the value
        return series



    def findEseries(self, W):
        '''
        For the given energy tolerance compute the necessary Es array

        Check the computed probabilities for (E[-1], W[1]) and (E[-2], W[1])
        are within tol_E of each other [i.e. probability of loosing the smallest
        amount of energy for an electron with max E is close within E_tol of the
        probability of loosing the same mount of energy for an electron with the
        next smaller energy]

        The tol_E aims to ensure the probabity of energy loss W as a function
        of E is relatively smooth function.
        '''

        # first compute the reference probability value at E[-1]=Emax
        E = self.Emax
        # simplify the excitation function to depend only on W
        integrandRef = lambda Wi: self.func(E, Wi, self.n_e)

        # trapezoidal intergral in the range [Wmin, W]
        smalldWInt = np.trapz([integrandRef(self.Wmin), integrandRef(W)],
                          x = [self.Wmin,               W              ] )

        # total W intergral at this E
        totalWInt,_ = quad(integrandRef, self.Wmin, maxW_moller(E, self.Ef), limit=300, epsabs=1e-24)

        # the probability is the fractional integral
        probEmax = smalldWInt/totalWInt


        # first refine for order of magnitude
        nE1, _ = self.findGeomSeries(startVal = self.Emin,
                                     endVal = self.Emax,
                                     maxSize = 100000,
                                     generator = ordersOfMag,
                                     test = self.check_probTol,
                                     refVal = probEmax,
                                     W = W)

        # second refine in multiple of 10% of the value
        _, series = self.findLinSeries(startVal = self.Emin,
                                        endVal = self.Emax,
                                        maxSize = nE1,
                                        generator = percents,
                                        test = self.check_probTol,
                                        refVal = probEmax,
                                        W=W)

        # and return the value
        return series


    def set_Ws(self):
        '''
        Set self.Ws = np.array(nW, )
        '''
        # do the checks for Emax
        E = self.Emax

        # find W series and set it
        self.Ws = self.findWseries(E)


    def set_Es(self):
        '''
        Set self.Es = np.array(nE, )
        '''

        # check if Ws was set
        if (self.Ws is None):
            print ('Must first set W series')

        # the range of W to consider
        #W = self.Ws[1]
        W = 100.
        # find E series and set it
        self.Es = self.findEseries(W)



    # def generate(self, target):
    #
    #
    #
    #
    # def bisect(self, ):










# test things

# set material
thisMaterial = material('Al')

Erange = (5000, 20000)
mollerTable = probTable('Moller', moller_dCS, Erange, 50, 5e-7, 1e-7, thisMaterial, 'test')
mollerTable.set_Ws()
print (len(mollerTable.Ws))
mollerTable.set_Es()
print (len(mollerTable.Es))
