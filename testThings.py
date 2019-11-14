import unittest
import random
import warnings

from multiprocessing import Process, cpu_count, Queue
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scattering import alpha, binaryCollModel, Rutherford_halfPol
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

def angleFreq(pdData, numBins, range):
    '''
    generate dataframe with frequency of angle appearance

    Parameters
    ----------
        pdData : pandas DataFrame
            containing two columns: labels and angles
        numBins : int
            number of bins
        range : tuple
            either (0, pi] or (0, 2pi]

    Returns
    -------
        pdFreq : pandas DataFrame
            containing the frequency of angles per bin
    '''
    bins = np.linspace(range[0], range[1], numBins+1)

    # add a frequency column
    pdData['freq'] = 1

    # make a new pandas df with binned angle frequencies
    pdFreq = pdData.groupby(['label', pd.cut(pdData.angles, bins)]).freq.sum().reset_index()

    # compute bin centers
    pdFreq['degrees'] = [np.degrees(interval.mid) for interval in pdFreq.angles]

    return pdFreq

def plotPolar_polar_px(pdData):
    ''' plotly express polar plot
    This is the fewest lines plot but I don't know how to set a range for theta
    '''
    print('/n', 'plotting...', '/n')

    numBins = 200

    # frequency data
    freq = angleFreq(pdData, numBins, (0, np.pi))

    # make polar figure plot MC results as scatter
    fig = px.line_polar(freq, r='freq', theta='degrees', line_dash='label',
                        color='label', log_r=True, template="plotly_dark",
                        start_angle=0, direction='counterclockwise')

    fig.show()

def plotPolar_polar_pltly(pdData):
    ''' plotly graph_objects polar plot'''
    print('\n', 'plotting...', '\n')

    numBins = 800

    # frequency data
    freq = angleFreq(pdData, numBins, (0, np.pi))

    # make figure
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'polar'}]*1]*1)

    # add theoretical line
    fig.add_trace( go.Scatterpolar( name = 'theory',
                    r = freq[freq.label=='theory']['freq'],
                    theta = freq[freq.label=='theory']['degrees'],
                    mode = 'lines'), 1, 1)

    # add MC scatter
    fig.add_trace( go.Scatterpolar(name = 'MC',
                    r = freq[freq.label=='MC']['freq'],
                    theta = freq[freq.label=='MC']['degrees'],
                    mode = 'markers' ), 1, 1)

    # show only [0, 180]
    fig.update_layout(
    title = 'Polar angular distributions',
    polar =dict(radialaxis = dict(tickangle = 45),
                sector = [0, 180]))

    fig.show()


def binOnCircle(angleList, nbins):
    ''' For a list of angles return a binned data in nbins

    Returns
    -------
        tuple: (radius, edges of theta)
    '''
    return np.histogram(angleList, bins=nbins, range=(0, 2*np.pi))

def binCenters(binEdgesList):
    ''' For a list of bin edges return the centers'''
    return list(map(np.mean, zip(binEdgesList[:-1], binEdgesList[1:])))

def thetaFromCos(Cos2HalfAngles):
    '''
    The angles are stored as cos^2(Theta/2). return the angles
    '''
    # these values should be withing arccos domain
    assert ( (all( np.array(Cos2HalfAngles)>=-1)) & (all(np.array(Cos2HalfAngles)<=1)) ), 'Some cos values are outside arccos domain'

    return  np.degrees(2*np.arccos(np.array(Cos2HalfAngles)**0.5))

def rndAnglesFromDF(scatter_data, scatter_type, angle_type, size, E = None):
    '''
    Pick a random sample of given size of scattering angles,
    either polar or azimuthal, for a given scattering type
    from the pandas dataframe

    Parameters
    ----------
    scatter_type : str
        'Rutherford', 'Moller', 'Gryzinski' or 'Quinn'
    angle_type : str
        'az_angle' or 'pol_angle'
    size : int
        size of sample
    E : float, optional
        scattering energy

    Returns
    -------
    angles :obj: `list` of float
        list of scattering angles
    '''
    if E:
        angles = random.sample(list(scatter_data[scatter_data.type==scatter_type][scatter_data.E==E][angle_type].values), size)
    else:
        angles = random.sample(list(scatter_data[scatter_data.type==scatter_type][angle_type].values), size)

    # these are saved as cos^2(Theta/2), return as angles
    return  thetaFromCos(angles)



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

    def watson_two_test(self, data, plot=False):
         '''
         Apply the Watson two test for two samples and check
         if the test statistics, U2, is larger than the critical value
         for p>0.05.

         Note: See page 151 in 'Directional statistics' K. V. Mardia and P. E. Jupp

         input :
                data : pandas table of angles containing theoretical and MC results
                plot : boolean
         '''
         if plot:
            # polar plot for polar angle
            plotPolar_polar_pltly(data)

         # pick only the first 100 points; this is good enough for the test
         data = data[0:100]

         # sort angles into x and y lists
         x = sorted(data[data.label=='theory'].angles.values)
         y = sorted(data[data.label=='MC'].angles.values)

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
    def TestPolarProb_Ruth_E0(self):
        '''
        Test if the polar angle distribution of Rutherford scatterings
        is within a certain range from the analytical probability distribution.

        This is done by applying the Watson two test for two samples, one taken
        from the simulated results and one from the analytical formulation and then
        checking if the test statistics, U2, is larger than the critical value
        with confidence 0.05 < p < 0.1

        Only for incident energy
        '''
        # size of sample
        n = 10000

        # read dataframe into pandas
        scatterings = pd.read_hdf(self.file, 'scatterings')

        # select only incident energy data
        E = np.array(self.inputPar['E0'])

        # choose n random values from the MC Rutherford polar scattering angles
        MCAngles = rndAnglesFromDF(scatterings, 'Rutherford', 'pol_angle', n, E)

        # make a pandas dataframe
        pdData = pd.DataFrame(data={'label':'MC', 'angles':MCAngles })

        # choose n random values from the analytical expression
        calculatedCos2HalfAngles = Rutherford_halfPol(E, self.material.params['Z'])
        calculatedAngles = thetaFromCos(calculatedCos2HalfAngles)

        # add to the dataframe
        pdData = pdData.append(pd.DataFrame(data={'label':'theory', 'angles':calculatedAngles }),
                            ignore_index=True)

        # are the two samples part of the same population?
        self.watson_two_test(pdData, True)


    def TestPolarProb_Ruth(self):
        '''
            Like above but for all energies.
        '''
        # size of sample
        n = 200000

        # read dataframe into pandas
        scatterings = pd.read_hdf(self.file, 'scatterings')

        # choose a random sample from the MC Rutherford polar scattering angles
        MCAngles =  rndAnglesFromDF(scatterings, 'Rutherford', 'pol_angle', n)

        # make a pandas dataframe
        pdData = pd.DataFrame(data={'label':'MC', 'angles':MCAngles })

        # choose n random values from the analytical expression
        # for a random sample of energies from the population of electron energies
        Elist = np.random.choice(scatterings.E, n)

        calculatedCos2HalfAngles = Rutherford_halfPol(Elist, self.material.params['Z'])
        calculatedAngles = thetaFromCos(calculatedCos2HalfAngles)

        # add to the dataframe
        pdData = pdData.append(pd.DataFrame(data={'label':'theory', 'angles':calculatedAngles }),
                            ignore_index=True)

        # are the two samples part of the same population?
        self.watson_two_test(pdData, True)


    #########################################################################
    ############################ Moller tests ###############################
    #########################################################################

    def TestPolarProb_Moller(self):
        '''
            Test if the polar angle distribution of Moller scatterings
            is within a certain range from the computed probability distribution
        '''
        # read dataframe into pandas
        scatterings = pd.read_hdf(self.file, 'scatterings')

        # choose 100 random values from the MC Moller polar scattering angles
        MCCos2HalfAngles = random.sample(list(scatterings[scatterings.type=='Moller']['pol_angle'].values), 1000)
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
    #suite.addTest(TestScatterAnglesforDS('TestPolarProb_Ruth_E0'))
    suite.addTest(TestScatterAnglesforDS('TestPolarProb_Ruth'))
    #suite.addTest(TestScatterAnglesforDS('TestPolarProb_Moller'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
