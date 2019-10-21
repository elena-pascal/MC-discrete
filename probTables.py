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


def binEdgePairs(inList):
    ''' For a given list, like the binEdges,
        return list of sets of two edges for each bin
    '''
    outList = [(inList[0], inList[0])]
    for i in range(len(inList)-1):
        outList.append((inList[i], inList[i+1]))
    return outList


def ordersOfMag(maxSize = 1000000, numIter = 5):
    ''' iterator over n = [100, 1000, 10000, 100000]'''
    #minSize = 100
    for n in np.geomspace(100, maxSize, num=numIter, dtype=int):
        yield int(n)


def percents(maxSize, numIter = 10):
    '''iterator over n = [n/10, 2*n/10 ... nW]'''
    for n in np.linspace(maxSize/10, maxSize, numIter, dtype=np.float32):
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

    def __init__(self, type, shell, func, E_range, tol_E, tol_W, mat, mapTarget, chunk_size, Wc=None ):
        '''
        Define the table

        type  = string; the type of table, Moller or Gryz

        func  = function; the excitation function for this type of scattering
        tol_E = float; relative difference in probabilities between two adiacent Es for same W
        tol_W = float; relative difference in probabilities between two adiacent Ws for same E

        The two tolerances are not independent of each other.
        '''
        self.type = type

        self.Ef   = mat.fermi_e

        self.shell = shell

        if(self.type == 'Moller'):
            # asign minimum allowed energy loss
            self.Wmin = Wc

            # Wmax is a function of E
            self.Wmax = lambda E: maxW_moller(E, self.Ef)

            # number of electrons in shell
            self.numEl = mat.params['n_val']

            # assign excitation function
            self.func_EW = lambda E, W: func(E, W, self.numEl) #f(E, W)

            # assign energy range
            self.Emin, self.Emax = E_range[0], E_range[1]

        elif('Gryz' in self.type):
            # asign minimum allowed energy loss
            self.Wmin =  mat.params['Es'][shell]

            # Wmax is a function of E
            self.Wmax = lambda E: maxW_gryz(E, self.Ef)

            # number of electrons in shell
            self.numEl = mat.params['ns'][shell]

            # assign excitation function
            self.func_EW = lambda E, W: func(E, W, self.numEl, self.Wmin)

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
        logging.info('difference between trapez integral on Wseries and quad is: %s', diff)

        # return the truth value
        return diff < self.tol_W


    def check_CDFTol(self, Eseries, refVal, integrand):
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
        logging.info('next bigger E: %s', E)

        # Energy loss position for which we test
        W = self.Ws[1]


        # trapezoidal intergral in the range [Wmin, W]
        smalldWInt = np.trapz([integrand(E, self.Wmin), integrand(E, W)],
                          x = [self.Wmin,               W         ] )

        # integral for the entire range [Wmin, Wmax]
        totalInt = np.trapz(integrand(E, self.Ws[self.Ws<=self.Wmax(E)]),
                                x = self.Ws[self.Ws<=self.Wmax(E)])

        # the probability is the fractional integral
        CDF = smalldWInt/totalInt
        logging.info('CDF: %s', totalInt)
        logging.info('ref CDF: %s', refVal)

        # relative difference; Note: CDF is normalised
        diff = abs(totalInt - refVal)/refVal

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
            logging.info('next iterator in geom series: %s', num)

            # choose nW integration points in log space
            # NOTE: type is set to float32 because geomspace has a small rouding error for
            # the last element which means Wseries[-1] could end up larger than Wmax
            # this should be fine for a number of bins up to 1e7
            series = np.geomspace(startVal, endVal, num=num, dtype=np.float32)

            if test_pass(series, refVal, integrand): # within tolerance
                withinTol = True

        return num, series


    @staticmethod
    def findLinSeries(startVal, endVal, maxSize, generator, test_pass, refVal, integrand):
        ''' refine the number of values along a given iterator and using
            given acceptance test'''

        withinTol = False
        iterator = generator(maxSize)

        while not withinTol:
            nE = next(iterator)

            # choose nE integration points in linear space
            Eseries = np.linspace(startVal, endVal, num=nE, dtype=np.float32)

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
        integrand = lambda W: self.func_EW(E, W)

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

        The tolerance sets how smooth the change from one energy to another is.
        This is done by comparing the total dW integral for two adiacent energy bins
        The energy bins are chosen to be Emax and the next smaller one.

        The tol_E aims to ensure the probabity of energy loss W as a function
        of E is relatively smooth function.
        '''

        # first compute the reference probability value at E[-1]=Emax
        Eref = self.Emax
        logging.info('Emax: %s', Eref)

        # excitation function for Eref
        integrandRef = lambda Wi: self.func_EW(Eref, Wi)

        # trapezoidal intergral in the range [Wmin, W]
        smalldWInt = np.trapz([integrandRef(self.Wmin), integrandRef(W)],
                          x = [self.Wmin,               W              ] )

        # total W intergral at this E. Absolute error must be smaller than integral*tolerance
        totalInt,_ = quad(integrandRef, self.Wmin, self.Wmax(Eref), limit=300, epsabs=1e-34)

        # the probability is the fractional integral
        probEmax = smalldWInt/totalInt
        logging.info('Reference probability is: %s', probEmax)

        # first refine for order of magnitude
        nE1, _ = self.findGeomSeries(startVal  = self.Emin,
                                     endVal    = self.Emax,
                                     maxSize   = 1000000,
                                     generator = ordersOfMag,
                                     test_pass = self.check_CDFTol,
                                     refVal    = totalInt,
                                     integrand = self.func_EW)

        # second refine in multiple of 10% of the value
        _, series = self.findLinSeries(startVal  = self.Emin,
                                       endVal    = self.Emax,
                                       maxSize   = nE1,
                                       generator = percents,
                                       test_pass = self.check_CDFTol,
                                       refVal    = totalInt,
                                       integrand = self.func_EW)

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
        assert (np.all(self.Ws < E)), 'Energy loss was found to be larger than the current electron energy!'


    def set_Es(self):
        '''
        Set self.Es = np.array(nE, )
        '''

        # check if Ws was set
        assert (self.Ws is not None), 'Set Ws first'

        W = self.Ws[1]
        logging.info('For W: %s, start finding Es', W)

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
            W_max_ar = maxW_gryz(E_block, self.Ef)

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
        funcs_EW_table = self.func_EW(E_mesh, W_mesh_masked)

        # stack the function table with its duplicate moved up by one cell to create fthe limits of the trapez of integration
        fun_vals = np.ma.stack((funcs_EW_table[:, :-1], funcs_EW_table[:, 1:]))

        # integrate the 2D table of functions between the two limits
        intStep = np.trapz(y=fun_vals, x=x_vals, axis=0)

        # cumulative sum on the stepwise integral
        cumInt_da = np.cumsum(intStep, axis=1)

        # compute cumulative distribution function table by dividing by cumInt_da[:, Wmax]
        CDF = cumInt_da/(np.broadcast_to(cumInt_da[:, -1], cumInt_da.T.shape).T)

        return CDF


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
        map computeBlock on the entire range of Es one block at a time
        '''
        print ()
        print ('Computing %s table for shell %s' %(self.type, self.shell))

        # set W series
        self.set_Ws()
        print ('Ws:', len(self.Ws))

        # set E series
        self.set_Es()
        print ('Es:', len(self.Es))

        # chunk Es to a dask array
        Es_da = da.from_array(self.Es, chunks = self.chunk_size)

        # actually compute table
        with ProgressBar():
            self.table = Es_da.map_blocks(func = self.computeBlock,
                            #chunks = (self.chunk_size, self.Ws.size-1 ),
                            dtype  = float ).compute()



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
                       shape    = (self.Es.size, self.Ws.size-1)   )


        # replace masked values with zero and save the table to the memory map
        map[:] = self.table.filled(0)[:]

        # flush memory changes to disk before removing the object
        del map
        print ('Table written to target:', self.target)

    def readFromMemory(self):
        '''
        Load the table from the memory map

        Note:
            See np.memmap for more info
        '''
        # set W series
        self.set_Ws()

        # set E series
        self.set_Es()

        readTable =  np.memmap(filename = self.target,
                       dtype    = 'float32',
                       mode     = 'r',
                       shape    = (self.Es.size, self.Ws.size-1)   )

        # mask zeros
        self.table = ma.masked_array(readTable, readTable==0)
        print ('read %s table for shell %s from memory' %(self.type, self.shell))
        print()



def genTables(inputPar):
    '''
    '''

    # define the Erange from input parameters
    Erange = (inputPar['Emin'], inputPar['E0'])

    # set chunk_size to whatever worked better on my machine
    csize = 100

    tables = {}

    materialInst = material(inputPar['material'])
    # instance for Moller table
    mollerTable = probTable(type='Moller', shell=materialInst.params['name_val'], func=moller_dCS,
                            E_range=Erange,
                            tol_E=inputPar['tol_E'], tol_W=inputPar['tol_W'],
                            mat=materialInst, mapTarget='tables', chunk_size=csize,
                            Wc=inputPar['Wc'])

    # generate Moller table
    #mollerTable.generate()

    # map to memory
    #mollerTable.mapToMemory()


    # read from disk
    mollerTable.readFromMemory()


    tables['Moller'] = mollerTable


    gTables_list = []
    # one Gryzinki table for each shell
    for Gshell in materialInst.params['name_s'][::-1]:
        # instance for Gryzinski table
        gryzTable = probTable(type='Gryzinski', shell=Gshell, func=gryz_dCS,
                            E_range=Erange,
                            tol_E=inputPar['tol_E'], tol_W=inputPar['tol_W'],
                            mat=materialInst, mapTarget='tables', chunk_size=csize)

        # generate Gryzinski table for shell Gshell
        #gryzTable.generate()

        # map to memory
        #gryzTable.mapToMemory()

        # read from disk
        gryzTable.readFromMemory()

        gTables_list.append(gryzTable)

    tables['Gryz'] = gTables_list


    return tables


    # elif (inputPar['mode'] in ['diel', 'dielectric']):
    #     print ' ---- calculating dielectric function integral table'
    #     tables_diel =





import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(filename='logs/tables.log',level=logging.INFO, filemode='w')
# test things
#
# # set material
# thisMaterial = material('Al')
#
# Erange = (5000., 20000.)
#
# func = moller_dCS
#
# mollerTable = probTable('Moller',thisMaterial.params['name_val'], func, Erange, 1e-4, 1e-7, thisMaterial, 'testData', 100, 50.)
# mollerTable.set_Ws()
# print ('Ws', len(mollerTable.Ws))
#
# mollerTable.set_Es()
# print ('Es', len(mollerTable.Es))
#
# mollerTable.generate()
# print ('Table was generated', mollerTable.table)
#
# mollerTable.mapToMemory()
# print ('Table was mapped to memory')
#
# mollerTable.readFromMemory()
# print ('Table was read from memory', mollerTable.table)



# func = gryz_dCS
#
# for shell in thisMaterial.params['name_s']:
#     gryzTable = probTable('Gryz',shell, func, Erange, 50., 1e-4, 1e-7, thisMaterial, 'testData', 100)
#     gryzTable.set_Ws()
#     print ('Ws', len(gryzTable.Ws))
#
#     gryzTable.set_Es()
#     print ('Es', len(gryzTable.Es))
#
#     gryzTable.generate()
#     print ('Tables were generated')
#
#     gryzTable.mapToMemory()
#     print ('Tables were mapped to memory')
#
#     gryzTable.readFromMemory()
#     print ('Tables were read from memory')
