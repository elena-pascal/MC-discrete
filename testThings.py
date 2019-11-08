import unittest
import random
import warnings

from multiprocessing import Process, cpu_count, Queue
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scattering import alpha, binaryCollModel
from multiScatter import scatterMultiEl_cont, scatterMultiEl_DS, retrieve
from material import material

from fileTools import readInput, thingsToSave, writeBSEtoHDF5
from probTables import genTables


def pol2cart(rho, phi):
    ' From polar coordinates to '
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def unique(inputList):
    ''' return a list of unique elements from list'''
    unique_list = []

    for value in inputList:
        # check if exists in unique_list already
        if value not in unique_list:
            unique_list.append(value)

    return unique_list

def az_polarPlot(plotDict):
    ''' matplotlib polar plot'''
    print()
    print('plotting...')
    print()

    #colorList = ['rgba(128, 177, 24 .9)', 'rgba(141, 211, 199 .9)']
    colorList = ['lightcoral', 'cadetblue']

    ax = plt.subplot(111, projection='polar')

    # plot theoretical results as a continuous line
    ax.plot(plotDict['theory']['thetas'], plotDict['theory']['radii'],
                alpha = 0.75, label='theory', color=colorList[0])

    # plot MC results as scatter
    ax.scatter(plotDict['Monte Carlo']['thetas'], plotDict['Monte Carlo']['radii'],
                alpha = 0.75, label='Monte Carlo', color=colorList[1])

    # plot only one hemisphere for azimuthal angle
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    # log?
    #ax.set_rlim((0.1, 1000))
    #ax.set_rscale('log')

    #ax.set_title(title, va='bottom')
    ax.legend()
    plt.show()

def binOnCircle(angleList, nbins):
    ''' For a list of angles return a binned data in nbins

    Return :
        tuple: (radius, edges of theta)
    '''
    return np.histogram(angleList, bins=nbins, range=(0, 2*np.pi))

def binCenters(binEdgesList):
    ''' For a list of bin edges return the centers'''
    return list(map(np.mean, zip(binEdgesList[:-1], binEdgesList[1:])))


# test the Monte Carlo results match the underlying probability ditribution from which events are sampled

class TestScatterAnglesforDS(unittest.TestCase):
    # def __init__(self, *args, **kwargs):
    #     super(TestScatterAnglesforDS, self).__init__(*args, **kwargs)
    #     self.inputPar = readInput('testData/testInput.file')
    #     self.material = material(self.inputPar['material'])
    #     self.tables  = genTables(self.inputPar)
    #     self.whatToSave = {'el_output':thingsToSave(self.inputPar['electron_output']),
    #                      'scat_output': thingsToSave(self.inputPar['scatter_output']) }
    #     self.file = 'testData/testAngles_DS.temp'

    def setUp(self):
        print ('Setting up...')
        print ()

        self.inputPar = readInput('testData/testInput.file')
        self.material = material(self.inputPar['material'])
        self.tables  = genTables(self.inputPar)
        self.whatToSave = {'el_output':thingsToSave(self.inputPar['electron_output']),
                         'scat_output': thingsToSave(self.inputPar['scatter_output']) }
        self.file = 'testData/testAngles_DS.temp'

        num_proc = cpu_count()-1
        output = {'electrons': Queue(), 'scatterings': Queue()}

        processes = [Process(target=scatterMultiEl_DS, args=(self.inputPar, self.tables,
                                            self.whatToSave, output, count)) for count in range(num_proc)]
        # start threads
        for p in processes:
            p.start()

        # get results from queue
        results = retrieve(processes, output)

        # wait for processes to end
        for p in processes:
            # wait until all processes have finished
            p.join()
            p.terminate()

        # save to file
        warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
        writeBSEtoHDF5(results, self.inputPar, self.file)

    def watson_two_test(self, x, y, plot=False, xLabel=None, yLabel=None):
         '''
         Apply the Watson two test for two samples and check
         if the test statistics, U2, is larger than the critical value
         for p>0.05.

         Note: See page 151 in 'Directional statistics' K. V. Mardia and P. E. Jupp

         input : x   : list, set 1 of observed measurements
                 y   : list, set 2 of observed measurements
                 plot: boolean
                 xLabel, yLabel : labels for x and y curves if plotted
         '''
         if plot:
            # polar plot
            x_radii, x_thetas = binOnCircle(np.radians(x), 3600)
            y_radii, y_thetas = binOnCircle(np.radians(y), 3600)

            plotDict = { xLabel:{'thetas':binCenters(x_thetas), 'radii':x_radii},
                         yLabel:{'thetas':binCenters(y_thetas), 'radii':y_radii} }

            az_polarPlot(plotDict)

         # sort x and y lists
         x, y = sorted(x), sorted(y)

         # number of observations in each set
         nx, ny = len(x), len(y)

         # total number of observations
         N = len(x)  + len(y)

         # ordered index lists starts at zero for no elements
         xi, yi = [0], [0]

         # dk initialised to null list
         dk = []

         # total list made out of elemts from both lists
         total = x + y

         # sorted and unique
         total = sorted(unique(total))

         for value in total:
             if value in x:
                 xi.append(x.index(value)+1)
             if value in y:
                 yi.append(y.index(value)+1)

             dk.append(xi[-1]/nx - yi[-1]/ny)

         dk2 = [value**2 for value in dk]

         # U2 formula
         # See eq 8.3.7, pg 151, in 'Directional Statistics' K. V. Mardia, P. E. Jupp
         U2 =  nx*ny * (sum(dk2) - sum(dk)**2/N)/N**2

         # crit value is 0.152 for a confidence 0.05 < p < 0.1
         # See Appendix 2.14 in 'Directional Statistics' K. V. Mardia, P. E. Jupp
         crit  = 0.152

         # assert if U2 < crit then the null hypothesis is not rejected
         # null hypothesis: the two sets of angles are not significantly different
         print ('U2:', U2)
         self.assertTrue (U2 < crit), 'The two sets of angles are not from the same distribution'


    #########################################################################
    ############################ Rutherford tests ###########################
    #########################################################################

    def TestAzimProb_Ruth_E0(self):
        '''
            Test if the azimuthal angle distribution of Rutherford scatterings
            is within a certain range from the analytical probability distribution.

            This is done by applying the Watson two test for two samples, one taken
            from the simulated results and one from the analytical formulation and then
            checking if the test statistics, U2, is larger than the critical value
            with confidence 0.05 < p < 0.1

            Only for incident energy
        '''
        # read dataframe into pandas
        scatterings = pd.read_hdf(self.file, 'scatterings')

        # select only incident energy data
        E = self.inputPar['E0']

        # choose 100 random values from the MC Rutherford azimuthal scattering angles
        MCCos2HalfAngles = random.sample(list(scatterings[scatterings.type=='Rutherford'][scatterings.E==E]['az_angle'].values), 100)
        MCAngles = np.degrees(2*np.arccos(np.array(MCCos2HalfAngles)**0.5))

        alphaR = alpha(E, self.material.params['Z'])

        # choose 100 random values from the analythical expression
        randomVals = np.array([random.random() for _ in range(100)])
        calculatedCosAngles = list(1 - (2 * alphaR * randomVals) /( 1 - alphaR - randomVals))
        calculatedAngles = np.degrees(np.arccos(np.array(calculatedCosAngles)))

        self.watson_two_test(list(MCAngles), list(calculatedAngles))


    def TestAzimProb_Ruth(self):
        '''
            Like above but for all energies.
        '''
        n = 10000

        # read dataframe into pandas
        scatterings = pd.read_hdf(self.file, 'scatterings')

        # choose 100 random values from the MC Rutherford azimuthal scattering angles
        MCCos2HalfAngles = random.sample(list(scatterings[scatterings.type=='Rutherford']['az_angle'].values), n)
        MCAngles = np.degrees(2*np.arccos(np.array(MCCos2HalfAngles)**0.5))

        # choose 100 random values from the analythical expression
        # for a random sample of energies from the population of electron energies
        Elist = np.random.choice(scatterings.E, n)

        alphaR = alpha(Elist, self.material.params['Z'])

        randomVals = np.array([random.random() for _ in range(n)])

        calculatedCosAngles = list(1 - (2 * alphaR * randomVals) /( 1 - alphaR - randomVals))
        calculatedAngles = np.degrees(np.arccos(np.array(calculatedCosAngles)))

        # are the two samples part of the same population?
        self.watson_two_test(list(MCAngles), list(calculatedAngles), True, 'Monte Carlo', 'theory')


    #########################################################################
    ############################ Moller tests ###############################
    #########################################################################

    def TestAzimProb_Moller(self):
        '''
            Test if the azimuthal angle distribution of Moller scatterings
            is within a certain range from the computed probability distribution
        '''
        # read dataframe into pandas
        scatterings = pd.read_hdf(self.file, 'scatterings')

        # choose 100 random values from the MC Rutherford azimuthal scattering angles
        MCCos2HalfAngles = random.sample(list(scatterings[scatterings.type=='Moller']['az_angle'].values), 1000)
        MCAngles = np.degrees(2*np.arccos(np.array(MCCos2HalfAngles)**0.5))

        # pick 100 random rows from the dataframe
        rows = scatterings.sample(n=1000)
        Elist, Wlist = rows.E, rows.E_loss

        calculatedCos2HalfAngles = binaryCollModel(Elist, Wlist)
        calculatedAngles = np.degrees(2*np.arccos(np.array(calculatedCos2HalfAngles)**0.5))

        # are the two samples part of the same population?
        self.watson_two_test(list(MCAngles), list(calculatedAngles), True, 'Monte Carlo', 'theory')


    #########################################################################
    ############################ Gryzinski tests ############################
    #########################################################################


########################################################################
######################### main ##########################################
#########################################################################
def suite():
    suite = unittest.TestSuite()
    #suite.addTest(TestScatterAnglesforDS('TestAzimProb_Ruth_E0'))
    suite.addTest(TestScatterAnglesforDS('TestAzimProb_Ruth'))
    #suite.addTest(TestScatterAnglesforDS('TestAzimProb_Moller'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
