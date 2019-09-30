import os.path as path
import sys
import numpy as np
from scipy.integrate import quad
import numpy.ma as ma
import logging

from dask import dataframe as dd
from dask import array as da
from dask import delayed, compute
from dask.diagnostics import ProgressBar

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

def maxW_gryz(E, Ef):
    '''
	return the lower limits of intergration of the gryzinski excitation function
	in:
        E  :: maximum energy to consider
        Ei :: binding energy of shell i
        Ef :: Fermi energy
	'''

    return  E - Ef




def ordersOfMag(maxSize = 1000000, numIter = 5):
    ''' iterator over n = [100, 1000, 10000, 100000]'''
    #minSize = 100
    for n in np.geomspace(100, maxSize, num=numIter, dtype=int):
        yield int(n)


def percents(maxSize, numIter = 10):
    '''iterator over n = [n/10, 2*n/10 ... nW]'''
    for n in np.linspace(maxSize/10, maxSize, numIter):
        yield int(n)





class probTable:
    ''' A probability table for Moller and Gryzinski
    energy losses

    P(E, W) = probability of W energy loss for an electron
              of incident energy E

    Format is:           E1 E2  ...  Einc
                        __________________
                   W1  | -- --  ...  --
                   W2  | -- --       --
                   .   |
                   .   |
                   .   | NaN
                   Wmax| NaN NaN ...

    The valid values in the table are just the top right triangle.
    '''

    def __init__(self, type, shell, func, E_range, Wc, tol_E, tol_W, material, mapTarget, chunk_size):
        '''
        Define the table

        type  = string; the type of table, Moller or Gryz

        func  = function; the excitation function for this type of scattering
        tol_E = float; relative difference in probabilities between two adiacent Es for same W
        tol_W = float; relative difference in probabilities between two adiacent Ws for same E

        The two tolerances are not independent of each other.
        '''
        self.type = type

        self.Ef   = material.fermi_e

        if(self.type == 'Moller'):
            # asign minimum allowed energy loss
            self.Wmin = Wc

            # Wmax is a function of E
            self.Wmax = lambda E: maxW_moller(E, self.Ef)

            # assign excitation function
            self.func = lambda W, E: func(E, W, material.params['n_val']) #f(E, W)

            # assign energy range
            self.Emin, self.Emax = E_range[0], E_range[1]

        elif('Gryz' in self.type):
            # asign minimum allowed energy loss
            self.Wmin =  material.params['Es'][shell]

            # Wmax is a function of E
            self.Wmax = lambda E: maxW_gryz(E, self.Ef)

            # assign excitation function
            self.func = lambda W, E: func(E, W, material.params['ns'][shell], self.Wmin)

            # check if Emin in the energy range was chosen to be above the binding energy
            if(E_range[0] < self.Wmin):
                # assign energy range
                self.Emin, self.Emax = self.Wmin, E_range[1]
            else:
                # assign energy range
                self.Emin, self.Emax = E_range[0], E_range[1]

        else:
            sys.exit('I did not understand the type of table')




        self.tol_E = tol_E
        self.tol_W = tol_W

        self.Es  = None
        self.Ws  = None

        self.table = None # dask 2D array for probabilities table

        self.target = path.join(mapTarget, str(type)+'_'+str(shell)+'.table')

        self.chunk_size = chunk_size


    def check_intTol(self, Wseries, refVal, integrand):
        ''' For a given Wseries, check if fine enough
        by looking at the trapezoidal integration error

        input:
            Wseries      : array with W series values
            refVal       : reference value for int(W), W=[Wseries[0]..Wseries[-1]]
            integrand(W) : function that is to be the integrand

        return:
            boolean for this Wseries being fine enough that the
            relative error between the trapezoidal integral and refVal (from scipy.quadrature)
            is less than the set tolerance
        '''

        # trapezoidal intergral on Wseries
        trapezInt = np.trapz(np.array([integrand(W) for W in Wseries]), x=Wseries)

        # relative difference
        diff = abs(trapezInt-refVal)/refVal
        logging.debug('difference between trapez integral on Wseries and quad is: %s', diff)

        # return the truth value
        return diff < self.tol_W


    def check_probTol(self, Eseries, refVal, integrand):
        ''' For a given Eseries, check if fine enough
        by looking how fast the probability of looking energy W changes

        input:
            Eseries        : array with E series values
            refVal         : reference value for prob(E, W)
            integrand(W,E) : function that is to be the integrand

        return:
            boolean for this Eseries being fine enough that the
            relative error between the trapezoidal integral and refVal
            is less than the set tolerance

        '''

        # energy position for which we test
        E = Eseries[-2] # second the largest energy

        # Energy loss position for which we test
        assert (self.Ws is not None), 'Set Ws first'
        W = self.Ws[10]

        # trapezoidal intergral in the range [Wmin, W]
        smalldWInt = np.trapz([integrand(self.Wmin, E), integrand(W, E)],
                          x = [self.Wmin,               W         ] )

        # gaussian quadrature integral for the entire range [Wmin, Wmax]
        totalInt, _ = quad(integrand, self.Wmin, self.Wmax(E), limit=300, epsabs=1e-30, args=(E,))

        # the probability is the fractional integral
        prob = smalldWInt/totalInt

        # relative difference
        diff = abs(prob - refVal)/refVal

        # return the truth value
        return diff < self.tol_E


    @staticmethod
    def findGeomSeries(startVal, endVal, maxSize, generator, test_pass, refVal, integrand):
        ''' refine the number of values along a given iterator and using
            given acceptance test'''

        withinTol = False
        iterator = generator(maxSize)

        while not withinTol:
            num = next(iterator)

            # choose nW integration points in log space
            Wseries = np.geomspace(startVal, endVal, num=num)

            if test_pass(Wseries, refVal, integrand): # within tolerance
                withinTol = True

        return num, Wseries


    @staticmethod
    def findLinSeries(startVal, endVal, maxSize, generator, test_pass, refVal, integrand):
        ''' refine the number of values along a given iterator and using
            given acceptance test'''

        withinTol = False
        iterator = generator(maxSize)

        while not withinTol:
            nE = next(iterator)

            # choose nE integration points in linear space
            Eseries = np.linspace(startVal, endVal, num=nE)

            if test_pass(Eseries, refVal, integrand): # within tolerance
                withinTol = True

        return nE, Eseries



    def findWseries(self, E):
        '''
        For the given energy loss tolerance compute the necessary Ws arrays

        We are computing the integrals in the table using the trapezoidal method (np.trapz)
        for speed. For low number of W values this will not be a good approximation since
        the integrand curves are logarithmic.

        Check that the computed integral is within tol_Int of simpy.quad

        input:
                E: float
        '''

        # compute Wmax
        W_max = self.Wmax(E)

        # simplify the excitation function to depend only on E and W
        integrand = lambda W: self.func(W, E)

        # gaussian quadrature integral
        quadInt, errorInt = quad(integrand, self.Wmin, W_max, limit=300, epsabs=1e-34)

        # check if the intergation error is not larger than the tollerance we allow for
        assert (errorInt < self.tol_W*quadInt),'The Ws tolarance is smaller than the error of integration'

        # first refine for order of magnitude
        nW1, _ = self.findGeomSeries(startVal = self.Wmin,
                                     endVal = W_max,
                                     maxSize = 1000000,
                                     generator = ordersOfMag,
                                     test_pass = self.check_intTol,
                                     refVal = quadInt,
                                     integrand = integrand)
        logging.info('in find W series, after first findGeomSeries the number of W bins is: %s', nW1)

        # second refine in multiple of 10% of the value
        nW2, series = self.findGeomSeries(startVal = self.Wmin,
                                        endVal = W_max,
                                        maxSize = nW1,
                                        generator = percents,
                                        test_pass = self.check_intTol,
                                        refVal = quadInt,
                                        integrand = integrand)
        logging.info('in find W series, after second findGeomSeries the number of W bins is: %s', nW2)

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
        Eref = self.Emax

        # excitation function for Eref
        integrandRef = lambda Wi: self.func(Wi, Eref)

        # trapezoidal intergral in the range [Wmin, W]
        smalldWInt = np.trapz([integrandRef(self.Wmin), integrandRef(W)],
                          x = [self.Wmin,               W              ] )

        # total W intergral at this E. Absolute error must be smaller than inegral*tolerance
        totalWInt,_ = quad(integrandRef, self.Wmin, maxW_moller(Eref, self.Ef), limit=300, epsabs=1e-34)

        # the probability is the fractional integral
        probEmax = smalldWInt/totalWInt

        # first refine for order of magnitude
        nE1, _ = self.findGeomSeries(startVal  = self.Emin,
                                     endVal    = self.Emax,
                                     maxSize   = 100000,
                                     generator = ordersOfMag,
                                     test_pass = self.check_probTol,
                                     refVal    = probEmax,
                                     integrand = self.func)

        # second refine in multiple of 10% of the value
        _, series = self.findLinSeries(startVal  = self.Emin,
                                       endVal    = self.Emax,
                                       maxSize   = nE1,
                                       generator = percents,
                                       test_pass = self.check_probTol,
                                       refVal    = probEmax,
                                       integrand = self.func)

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

        # test if W are still smaller than E
        assert (self.Ws.any() < E), 'Energy loss was found to be larger than the current electron energy!'


    def set_Es(self):
        '''
        Set self.Es = np.array(nE, )
        '''

        # check if Ws was set
        if (self.Ws is None):
            print ('Should maybe set W series first.')
            W = 100. # use an aleatory but small energy loss for testing
        else:
            #the range of W to consider
            W = self.Ws[10]

        # find E series and set it
        self.Es = self.findEseries(W)



    def computeBlock(self, E_block):
        '''
        Compute dask 2D array with probabilities for a given energy range block
        '''

        if self.Ws is None:
            # set Ws series
            self.set_Ws()

        if self.Es is None:
            # set Es series
            self.set_Es()

        # set max value for Ws
        if (self.type == 'Moller'):
            W_max_ar = maxW_moller(E_block, self.Ef)
        elif ('Gryz' in self.type):
            W_max_ar = maxW_gryz(E_block, self.Wmin, self.Ef)

        # make a mesh of E_list and W_list
        E_mesh, W_mesh = np.meshgrid(E_block, self.Ws, indexing='ij')

        # mask values larger than W_max(E)
        # suppress operation on that side of the table
        invalidMask = W_mesh > np.broadcast_to(W_max_ar, W_mesh.T.shape).T

        # online compute things on the masked W mesh
        W_mesh_masked = ma.masked_array(W_mesh, invalidMask)

        # make a 3D array with the limits of integration
        x_vals = np.ma.stack((W_mesh_masked[:, :-1], W_mesh_masked[:, 1:]))

        # function value at every step - 2D table
        funcs_EW_table = self.func(W_mesh_masked, E_mesh)

        # stack the function table with its duplicate moved up by one cell to create the limits of the trapez of integration
        fun_vals = np.ma.stack((funcs_EW_table[: , :-1], funcs_EW_table[:, 1:]))

        # integrate the 2D table of functions between the two limits
        intStep = np.trapz(y=fun_vals, x=x_vals, axis=0)

        # cumulative sum on the stepwise integral
        cumInt_da = np.cumsum(intStep, axis=1)

        # compute probability table
        prob = cumInt_da/(np.broadcast_to(cumInt_da[:, -1], cumInt_da.T.shape).T)

        return prob


    # def toParquet(self, target):
    #     #put dataframe into parquet
    #     self.ddf.to_parquet(target, has_nulls=False, compute=True)


    # def oneBlock(self,E_block):
    #     '''
    #     input:
    #         Ein: energy value, array
    #
    #     output:
    #         prob = cumsum/max(cumsum) of the integral for that E
    #
    #     output is transformed to dask dataframe and saved to parquet
    #     '''
    #
    #     #compute dataframe
    #     dask_df = self.compute_ddf(E_block)
    #
    #     target = os.path.join(self.target, str(int(E_block[0])))
    #     #put dataframe into parquet
    #     self.toParquet(target)
    #
    #     return

    def generate(self):
        '''
        map genBlock on the entire range of Es one block at a time
        '''
        # chunk Es to a dask array
        Es_da = da.from_array(self.Es, chunks = (self.chunk_size))

        print ()
        print ('Computing', self.type, 'table:')
        with ProgressBar():
            self.table = Es_da.map_blocks(func  = self.computeBlock,
                            chunks = (self.Es.size/self.chunk_size, self.Ws.size ),
                            dtype  = float).compute()
        print ()


    def mapToMemory(self):
        '''
        Create a memory map to the table stored in a binary file on disk
        The binary file is the class target

        Note:
            See np.memmap for more info
        '''
        # create a memory map with defined dtype and shape
        map = np.memmap(filename = self.target,
                       dtype    = 'float32',
                       mode     = 'w+',
                       shape    = (self.Es.size, self.Ws.size)   )

        # write data to the memory map
        map = self.table

        # flush memory changes to disk before removing the object
        del map


    def readFromMemory(self):
        '''
        Load the table from the memory map

        Note:
            See np.memmap for more info
        '''
        self.table =  np.memmap(filename = self.target,
                       dtype    = 'float32',
                       mode     = 'r',
                       shape    = (self.Es.size, self.Ws.size)   )

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# test things

# set material
thisMaterial = material('Al')

Erange = (5000, 20000)
# 
# func = moller_dCS
#
# mollerTable = probTable('Moller','foo', func, Erange, 50, 5e-7, 1e-7, thisMaterial, 'testData/Moller.dat', 100)
# mollerTable.set_Ws()
# print ('Ws', len(mollerTable.Ws))
#
# mollerTable.set_Es()
# print ('Es', len(mollerTable.Es))
#
# mollerTable.generate()
# print ('Table was generated')
#
# mollerTable.mapToMemory()
# print ('Table was mapped to memory')
#
# mollerTable.readFromMemory()
# print ('Table was read from memory', type(mollerTable.table))

logging.basicConfig(filename='tables.log',level=logging.INFO)

func = gryz_dCS

for shell in thisMaterial.params['name_s']:
    gryzTable = probTable('Gryz',shell, func, Erange, 50, 5e-4, 1e-4, thisMaterial, 'testData', 100)
    gryzTable.set_Ws()
    print ('Ws', len(gryzTable.Ws))

    gryzTable.set_Es()
    print ('Es', len(gryzTable.Es))

    gryzTable.generate()
    print ('Table was generated')

    gryzTable.mapToMemory()
    print ('Table was mapped to memory')

    gryzTable.readFromMemory()
    print ('Table was read from memory')
