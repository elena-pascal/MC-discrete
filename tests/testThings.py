import unittest
import random
import warnings

from multiprocessing import Process, cpu_count, Queue
import pandas as pd
import numpy as np
from scipy import integrate, stats

#import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from crossSections import ruther_sigma, moller_sigma, gryz_sigma, quinn_sigma
from crossSections import Ruth_diffCS, Ruth_diffCS_E
from extFunctions import Moller_W_E, Moller_W, Gryz_W_E, Gryz_W

from scattering import alpha, binaryCollModel, Rutherford_halfPol
from multiScatter import scatterMultiEl_cont, scatterMultiEl_DS, retrieve
from material import material

from fileTools import readInput, thingsToSave, writeBSEtoHDF5
from probTables import genTables, maxW_moller, maxW_gryz


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
            either (0, 180] or (0, 360]

    Returns
    -------
        pdFreq : pandas DataFrame
            containing the frequency of angles per bin
    '''
    if (range == (0,180)):
        # for polar angle distirbution most change happens close to zero
        bins = np.geomspace(range[0]+0.01, range[1], num=numBins)
    else:
        # for azimuthal angle we're looking at a uniform distribution
        bins = np.linspace(range[0], range[1], num=numBins)

    # add a frequency column
    pdData['freq'] = 1

    # make a new pandas df with binned angle frequencies
    pdFreq = pdData.groupby(['label', pd.cut(pdData.angles_deg, bins)]).freq.sum().reset_index()

    # compute bin centers
    pdFreq['degrees'] = [interval.mid for interval in pdFreq.angles_deg]

    return pdFreq

def plotPolar_polar_px(pdData, numBins=100):
    ''' plotly express polar plot
    This is the fewest lines plot but I don't know how to set a range for theta
    '''
    print('\n', 'plotting...', '\n')

    # frequency data
    freq = angleFreq(pdData, numBins, (0, 180))

    # make polar figure plot MC results as scatter
    fig = px.line_polar(freq, r='freq', theta='degrees', line_dash='label',
                        color='label', log_r=True, template="plotly_dark",
                        start_angle=0, direction='counterclockwise')

    fig.show()

def plotPolar_pltly(pdData, numBins, angle_type='polar'):
    ''' plotly graph_objects polar plot'''
    print('\n', 'plotting...', '\n')

    if (angle_type is 'polar'):
        # polar range is [0,180)
        range = (0, 180)

        # make r-axis log since it is very rapidly changing
        scaleType = 'log'
    else:
        # azimuthal range is [0,360)
        range = (0, 360)

        # keep r-axis linear
        scaleType = 'linear'

    # frequency data
    freq = angleFreq(pdData, numBins, range)

    # make figure
    #fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'polar'}]*1]*1)
    fig = go.Figure()

    # add theoretical line
    fig.add_trace( go.Scatterpolar( name = 'theory',
                                    r = freq[freq.label=='theory']['freq'],
                                    theta = freq[freq.label=='theory']['degrees'],
                                    mode = 'lines'))

    # add MC scatter
    fig.add_trace( go.Scatterpolar(name = 'MC',
                                    r = freq[freq.label=='MC']['freq'],
                                    theta = freq[freq.label=='MC']['degrees'],
                                    mode = 'markers' ))

    # show only [0, 180] for polar and all 360 for azimuthal
    fig.update_layout(
        title = angle_type + ' angular distributions',
        polar = dict(radialaxis = dict(tickangle = 45, type=scaleType),
                sector = [range[0], range[1]]))

    fig.show()


def binOnCircle(angleList, nbins):
    ''' For a list of angles return a binned data in nbins

    Returns
    -------
        tuple: (radius, edges of theta)
    '''
    return np.histogram(angleList, bins=nbins, range=(0, 2*180))

def binCenters(binEdgesList):
    ''' For a list of bin edges return the centers'''
    return list(map(np.mean, zip(binEdgesList[:-1], binEdgesList[1:])))

def thetaFromCos(Cos2HalfAngles):
    '''
    The angles are stored as cos^2(Theta/2). return the angles in degrees
    '''
    # these values should be withing arccos domain
    assert ( (all( np.array(Cos2HalfAngles)>=-1)) & (all(np.array(Cos2HalfAngles)<=1)) ),\
                        'Some cos values are outside arccos domain'

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
        list of scattering angles in degrees
    '''
    if E:
        print(scatter_data[(scatter_data.type.str.contains(scatter_type, regex=False ))& (scatter_data.E==E)])
        angles = random.sample(list(scatter_data[(scatter_data.type.str.contains(scatter_type, regex=False ))
                                        & (scatter_data.E==E)][angle_type].values), size)
    else:
        angles = random.sample(list(scatter_data[(scatter_data.type.str.contains(scatter_type, regex=False ))][angle_type].values), size)

    if (angle_type is 'pol_angle'):
        # these are saved as cos^2(Theta/2), return as angles
        return  thetaFromCos(angles)
    else :
        # azimuthal angles are saved as (phi/2)
        return np.degrees(np.array(angles)*2)



##############################################################################
######          Scatter angles                                          ######
##############################################################################
class TestScatterAnglesforDS(unittest.TestCase):
    '''
    test the Monte Carlo results match the underlying probability ditribution
    from which events are sampled
    '''

    def setUp(self):
        print ('Setting up...', '\n')

        self.inputPar = readInput('testData/testInput.file')
        self.material = material(self.inputPar['material'])
        self.tables  = genTables(self.inputPar)
        self.whatToSave = {'el_output':thingsToSave(self.inputPar['electron_output']),
                         'scat_output': thingsToSave(self.inputPar['scatter_output']) }
        self.file = 'testData/testAngles_DS.temp'

        self.num_proc = cpu_count()-1

        output = {'electrons': Queue(), 'scatterings': Queue()}

        processes = [Process(target=scatterMultiEl_DS, args=(self.inputPar, self.tables,
                                            self.whatToSave, output, count)) for count in range(self.num_proc)]
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

    # @classmethod
    # def setUpClass(cls):
    #     """ get_some_resource() is slow, to avoid calling it for each test use setUpClass()
    #         and store the result as class variable
    #     """
    #     super(TestScatterAnglesforDS, cls).setUpClass()
    #     cls.MCdata = generateData()

    def watson_two_test(self, data, angle_type, plot=False, numBins=30):
         '''
         Apply the Watson two test for two samples and check
         if the test statistics, U2, is larger than the critical value
         for p>0.05.

         Note: See page 151 in 'Directional statistics' K. V. Mardia and P. E. Jupp

         input :
                data    : pandas table of angles containing theoretical and MC results
                plot    : boolean
                numBins : bins for plot
         '''
         if plot:
            # polar plot for polar angle
            plotPolar_pltly(data, numBins, angle_type)

         # pick only the first 100 points for each label; this is good enough for the test
         # sort angles in degrees into x and y lists
         x = sorted(data[data.label=='theory'].iloc[0:200].angles_deg.values)
         y = sorted(data[data.label=='MC'].iloc[0:200].angles_deg.values)

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
         print ('U2:', U2, 'crit:', crit)
         self.assertTrue (U2 < crit), 'The two sets of angles are not from the same distribution'

    #########################################################################
    ############################ cross section tests ########################
    #########################################################################
    def Test_crossSection(self):
        '''
        Is the probability of different channels scattering matching
        the theoretical cross section at incident energy?
        '''
        print ('\n', 'Cross sections at E0',
               '\n',  '--------------------------------------')

        # size of sample must be at least 500 for stats to work out
        # if this is larger than the number of events at E0 than we oversample
        n = 500

        # select only incident energy data
        E = np.array(self.inputPar['E0'])

        ################ expected #############################################
        sigmas = {}
        # Rutherford CS at energy E0
        sigmas['p_R'] = ruther_sigma(E, self.material.params['Z'])

        # Moller CS at energy E0
        sigmas['p_M'] = moller_sigma(E, self.inputPar['Wc'], self.material.params['n_val'])

        # Gryzinski labels
        Gryzinski = [i+j for i,j in zip(['Gryzinski']*len(self.material.params['name_s']), self.material.params['name_s'])]

        # one Gryzinski CS for each inner shell
        for key in self.material.params['name_s']:
            sigmas['p_G'+key] = gryz_sigma(E, self.material.params['Es'][key], self.material.params['ns'][key])

        # Quinn CS at energy E0
        sigmas['p_Q'] = quinn_sigma(E, self.material.plasmon_e, self.material.fermi_e, self.material.atnd)

        normalisation = sum(sigmas.values())
        # sample n values from the theoretical cross sections
        expected = np.random.choice(['Rutherford', 'Moller', *Gryzinski, 'Quinn'], n,
                                    p=np.array(list(sigmas.values()))/normalisation)

        # make a dataframe with the counts; last bit removes redundant multilevel index
        exp_df = pd.crosstab(index=expected, columns=['count'], colnames=['type']).rename_axis(None)

        ################### observed ##########################################
        # read dataframe into pandas
        scatterings = pd.read_hdf(self.file, 'scatterings')

        # choose n random scattering types from the MC results
        observed = np.random.choice(scatterings[scatterings.E==E]['type'].values, n)

        # make it a pandas categorical to ensure we have all types even if we didn't sample some of them
        observed = pd.Categorical(observed, categories=exp_df.index.values)

        # make a dataframe with the counts; last bit removes redundant multilevel index
        obs_df = pd.crosstab(index=observed, columns=['count'], colnames=['type'],
                                        dropna=False).rename_axis(None)

        # chi square statistic for observed data is the same as the expected distribution
        chi2stat, _ = stats.chisquare(f_obs=obs_df, f_exp=exp_df)

        # critical chi square statistics value
        crit = stats.chi2.ppf(q=0.95,              # at 95% confidence
                            df= len(exp_df)-1  ) # degrees of freedom

        # reject null hypothesis that the two distributions are the same
        # if chi2stat > crit
        self.assertTrue (chi2stat < crit), 'The observed scattering types probabilities do not match those expected'


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
        print ('\n', 'Rutherford polar angle scattering at E0',
               '\n',  '--------------------------------------')

        # size of sample is equal to number of electrons scattered
        n = self.inputPar['num_el']

        # read dataframe into pandas
        scatterings = pd.read_hdf(self.file, 'scatterings')

        # select only incident energy data
        E = np.array(self.inputPar['E0'])

        # choose n random values from the MC Rutherford polar scattering angles
        MCAngles = rndAnglesFromDF(scatterings, 'Rutherford', 'pol_angle', n, E)

        # add these angles in degrees to a pandas dataframe
        pdData = pd.DataFrame(data={'label':'MC', 'angles_deg':MCAngles })

        # make a probability distribution for Rutherford angular scattering
        Ruth_dist = Ruth_diffCS_E(a=0, b=180, Z=self.material.params['Z'])

        # pick n angles from this distribution
        thAngles = Ruth_dist.rvs(size=n, E=E)

        # add to the dataframe
        pdData = pdData.append(pd.DataFrame(data={'label':'theory', 'angles_deg':thAngles }),
                            ignore_index=True)

        # are the two samples part of the same population?
        self.watson_two_test(pdData, 'polar', True)


    def TestPolarProb_Ruth(self):
        '''
            Like above but for all energies.
        '''
        print ('\n', 'Rutherford polar angle scattering at all Es',
               '\n',  '------------------------------------------')

        # size of sample
        n = self.inputPar['num_el']*self.num_proc

        # read dataframe into pandas
        scatterings = pd.read_hdf(self.file, 'scatterings')

        # choose a random sample from the MC Rutherford polar scattering angles
        MCAngles =  rndAnglesFromDF(scatterings, 'Rutherford', 'pol_angle', n)

        # make a pandas dataframe
        pdData = pd.DataFrame(data={'label':'MC', 'angles_deg':MCAngles })

        # choose 10*n energy values
        Elist = np.random.choice(scatterings.E, 10*n)

        # array of bins
        bins = np.linspace(start=self.inputPar['Emin'], stop=self.inputPar['E0'],
                            endpoint=True, num=100)


        # get a histrogram from the energy list
        E_weight, _ = np.histogram(Elist, bins = bins)

        # probability mass function for E list
        E_weight = E_weight/n

        # energy values are
        E_bins = binCenters(bins)

        # put these in a DataFrame
        E_df = pd.DataFrame({'energy':E_bins, 'weight':E_weight})

        # instance of probability ditribution at this sample of angles
        Ruth_dist = Ruth_diffCS(a=0, b=180, Z=self.material.params['Z'], Edist_df=E_df)

        # theoretical angles
        print ('\n', '...sampling from theoretical distribution', '\n')
        thAngles = Ruth_dist.rvs(size=n)

        # add to the dataframe
        pdData = pdData.append(pd.DataFrame(data={'label':'theory', 'angles_deg':thAngles }),
                            ignore_index=True)

        # are the two samples part of the same population?
        self.watson_two_test(pdData, 'polar', True, 50)


    def TestAzimProb_Ruth(self):
        '''
            Test for the azimuthal angle
        '''
        print ('\n', 'Rutherford azimuthal angle scattering at all Es',
               '\n', '-----------------------------------------------' )

        # size of sample
        n = self.inputPar['num_el']*self.num_proc

        # read dataframe into pandas
        scatterings = pd.read_hdf(self.file, 'scatterings')

        # choose a random sample from the MC Rutherford polar scattering angles
        MCAngles =  rndAnglesFromDF(scatterings, 'Rutherford', 'az_angle', n)

        # make a pandas dataframe
        pdData = pd.DataFrame(data={'label':'MC', 'angles_deg':MCAngles })


        # theoretical angles
        print ('\n', '...sampling from theoretical distribution', '\n')

        # the theoretical azimuthal angle is just a random number in [0, 360)
        thAngles = np.random.random_sample((n))*360

        # add to the dataframe
        pdData = pdData.append(pd.DataFrame(data={'label':'theory', 'angles_deg':thAngles }),
                            ignore_index=True)

        # are the two samples part of the same population?
        self.watson_two_test(pdData, 'azimuthal', True, 50)


    #########################################################################
    ############################ Moller tests ###############################
    #########################################################################

    def TestPolarProb_Moller_E0(self):
        '''
            Test if the polar angle distribution of Moller scatterings
            is within a certain range from the computed probability distribution
        '''
        print ('\n', 'Moller polar angle scattering at E0',
               '\n', '----------------------------------')

        # size of sample is equal to number of electrons scattered
        n = int(self.inputPar['num_el']*0.5)

        # read dataframe into pandas
        scatterings = pd.read_hdf(self.file, 'scatterings')

        # select only incident energy data
        E = np.array(self.inputPar['E0'])

        # choose n random values from the MC Moller polar scattering angles
        MCAngles = rndAnglesFromDF(scatterings, 'Moller', 'pol_angle', n, E)

        # add these angles in degrees to a pandas dataframe
        pdData = pd.DataFrame(data={'label':'MC', 'angles_deg':MCAngles })

        # the left end of the prob distribution is the def of max W
        b = maxW_moller(E, self.material.fermi_e)

        # make a probability distribution for Moller angular scattering
        Moller_W_dist = Moller_W_E(a=self.inputPar['Wc'], b=b, xtol=1e-3,
                                    nfree=self.material.params['n_val'],
                                    Ef=self.material.fermi_e)

        # pick n Ws from this distribution
        Ws = Moller_W_dist.rvs(size=n, E=E)

        # compute corresponding angles in degrees
        thAngles = thetaFromCos(binaryCollModel(E, Ws))

        # add to the dataframe
        pdData = pdData.append(pd.DataFrame(data={'label':'theory', 'angles_deg':thAngles }),
                            ignore_index=True)

        # are the two samples part of the same population?
        self.watson_two_test(pdData, 'polar', True, 50)

    def TestPolarProb_Moller(self):
        '''
            Like above but for all energies.
        '''
        # size of sample
        n = self.inputPar['num_el']*self.num_proc

        # read dataframe into pandas
        scatterings = pd.read_hdf(self.file, 'scatterings')

        # choose a random sample from the MC Moller polar scattering angles
        MCAngles =  rndAnglesFromDF(scatterings, 'Moller', 'pol_angle', n)

        # make a pandas dataframe
        pdData = pd.DataFrame(data={'label':'MC', 'angles_deg':MCAngles })

        # choose 10*n energy values
        Elist = np.random.choice(scatterings.E, 10*n)

        # array of bins
        bins = np.linspace(start=self.inputPar['Emin'], stop=self.inputPar['E0']/2,
                            endpoint=True, num=100)


        # get a histrogram from the energy list
        E_weight, _ = np.histogram(Elist, bins = bins)

        # probability mass function for E list
        E_weight = E_weight/n

        # energy values are
        E_bins = binCenters(bins)

        # put these in a DataFrame
        E_df = pd.DataFrame({'energy':E_bins, 'weight':E_weight})

        # the left end of the prob distribution is the def of max W
        b = maxW_moller(self.inputPar['E0'], self.material.fermi_e)

        # instance of probability ditribution at this sample of angles
        Moller_W_dist = Moller_W(a=self.inputPar['Wc'], b=b,
                                nfree=self.material.params['n_val'], Edist_df=E_df,
                                Ef=self.material.fermi_e)

        print ('\n', '...sampling from theoretical distribution', '\n')
        # pick n Ws from this distribution
        Ws = Moller_W_dist.rvs(size=n)

        # compute corresponding angles in degrees
        thAngles = thetaFromCos(binaryCollModel(E, Ws))

        # add to the dataframe
        pdData = pdData.append(pd.DataFrame(data={'label':'theory', 'angles_deg':thAngles }),
                            ignore_index=True)

        # are the two samples part of the same population?
        self.watson_two_test(pdData, 'polar', True, 50)



    def TestAzimProb_Moller(self):
        '''
            Test for the azimuthal angle
        '''
        print ('\n', 'Moller azimuthal angle scattering at all Es',
               '\n', '-----------------------------------------------' )

        # size of sample
        n = self.inputPar['num_el']*self.num_proc

        # read dataframe into pandas
        scatterings = pd.read_hdf(self.file, 'scatterings')

        # choose a random sample from the MC Moller polar scattering angles
        MCAngles =  rndAnglesFromDF(scatterings, 'Moller', 'az_angle', n)

        # make a pandas dataframe
        pdData = pd.DataFrame(data={'label':'MC', 'angles_deg':MCAngles })


        # theoretical angles
        print ('\n', '...sampling from theoretical distribution', '\n')

        # the theoretical azimuthal angle is just a random number in [0, 360)
        thAngles = np.random.random_sample((n))*360

        # add to the dataframe
        pdData = pdData.append(pd.DataFrame(data={'label':'theory', 'angles_deg':thAngles }),
                            ignore_index=True)

        # are the two samples part of the same population?
        self.watson_two_test(pdData, 'azimuthal', True, 50)



    #########################################################################
    ############################ Gryzinski tests ############################
    #########################################################################

    def TestPolarProb_Gryz_E0(self):
        '''
            Test if the polar angle distribution of Moller scatterings
            is within a certain range from the computed probability distribution
        '''
        print ('\n', 'Gryzinski polar angle scattering at E0',
               '\n', '----------------------------------')

        # size of sample is equal to number of electrons scattered
        n = self.inputPar['num_el']

        # read dataframe into pandas
        scatterings = pd.read_hdf(self.file, 'scatterings')

        # select only incident energy data
        E = np.array(self.inputPar['E0'])

        # choose n random values from the MC Gryzinski polar scattering angles
        MCAngles = rndAnglesFromDF(scatterings, 'Gryzinski', 'pol_angle', n, E)

        print ('Mc angles', MCAngles)
        # add these angles in degrees to a pandas dataframe
        pdData = pd.DataFrame(data={'label':'MC', 'angles_deg':MCAngles })

        # the right end of the prob distribution is min of binding energies of inner shells
        a = min(list(self.material.params['Es'].values()))

        # the left end of the prob distribution is the def of max W
        b = maxW_gryz(E, self.material.fermi_e)

        # make a probability distribution for Gryzinski angular scattering
        Gryz_W_dist = Gryz_W_E(a=a, b=b, xtol=1e-3,
                                    nsi=self.material.params['ns'],
                                    Ebi=self.material.params['Es'],
                                    Ef=self.material.fermi_e)

        # pick n Ws from this distribution
        Ws = Gryz_W_dist.rvs(size=n, E=E)

        # compute corresponding angles in degrees
        thAngles = thetaFromCos(binaryCollModel(E, Ws))
        print ('th angles', thAngles)
        # add to the dataframe
        pdData = pdData.append(pd.DataFrame(data={'label':'theory', 'angles_deg':thAngles }),
                            ignore_index=True)

        # are the two samples part of the same population?
        self.watson_two_test(pdData, 'polar', True, 50)

    def TestPolarProb_Gryz(self):
        '''
            Like above but for all energies.
        '''
        # size of sample
        n = self.inputPar['num_el']*self.num_proc

        # read dataframe into pandas
        scatterings = pd.read_hdf(self.file, 'scatterings')

        # choose a random sample from the MC Gryzinski polar scattering angles
        MCAngles =  rndAnglesFromDF(scatterings, 'Gryz', 'pol_angle', n)

        # make a pandas dataframe
        pdData = pd.DataFrame(data={'label':'MC', 'angles_deg':MCAngles })

        # choose 10*n energy values
        Elist = np.random.choice(scatterings.E, 10*n)

        # array of bins
        bins = np.linspace(start=self.inputPar['Emin'], stop=self.inputPar['E0']/2,
                            endpoint=True, num=100)


        # get a histrogram from the energy list
        E_weight, _ = np.histogram(Elist, bins = bins)

        # probability mass function for E list
        E_weight = E_weight/n

        # energy values are
        E_bins = binCenters(bins)

        # put these in a DataFrame
        E_df = pd.DataFrame({'energy':E_bins, 'weight':E_weight})

        # the left end of the prob distribution is the def of max W
        b = maxW_moller(self.inputPar['E0'], self.material.fermi_e)

        # instance of probability ditribution at this sample of angles
        Gryz_W_dist = Moller_W(a=self.inputPar['Wc'], b=b,
                                nfree=self.material.params['n_val'], Edist_df=E_df,
                                Ef=self.material.fermi_e)

        print ('\n', '...sampling from theoretical distribution', '\n')
        # pick n Ws from this distribution
        Ws = Gryz_W_dist.rvs(size=n)

        # compute corresponding angles in degrees
        thAngles = thetaFromCos(binaryCollModel(E, Ws))

        # add to the dataframe
        pdData = pdData.append(pd.DataFrame(data={'label':'theory', 'angles_deg':thAngles }),
                            ignore_index=True)

        # are the two samples part of the same population?
        self.watson_two_test(pdData, 'polar', True, 50)



    def TestAzimProb_Gryz(self):
        '''
            Test for the azimuthal angle
        '''
        print ('\n', 'Gryzinski azimuthal angle scattering at all Es',
               '\n', '-----------------------------------------------' )

        # size of sample
        n = self.inputPar['num_el']*self.num_proc

        # read dataframe into pandas
        scatterings = pd.read_hdf(self.file, 'scatterings')

        # choose a random sample from the MC Rutherford polar scattering angles
        MCAngles =  rndAnglesFromDF(scatterings, 'Gryz', 'az_angle', n)

        # make a pandas dataframe
        pdData = pd.DataFrame(data={'label':'MC', 'angles_deg':MCAngles })


        # theoretical angles
        print ('\n', '...sampling from theoretical distribution', '\n')

        # the theoretical azimuthal angle is just a random number in [0, 360)
        thAngles = np.random.random_sample((n))*360

        # add to the dataframe
        pdData = pdData.append(pd.DataFrame(data={'label':'theory', 'angles_deg':thAngles }),
                            ignore_index=True)

        # are the two samples part of the same population?
        self.watson_two_test(pdData, 'azimuthal', True, 50)

########################################################################
######################### main ##########################################
#########################################################################
def suite():
    suite = unittest.TestSuite()
    #suite.addTest(TestScatterAnglesforDS('TestPolarProb_Ruth_E0'))
    #suite.addTest(TestScatterAnglesforDS('TestPolarProb_Ruth'))
    #suite.addTest(TestScatterAnglesforDS('TestAzimProb_Ruth'))
    #suite.addTest(TestScatterAnglesforDS('TestPolarProb_Moller_E0'))
    #suite.addTest(TestScatterAnglesforDS('TestPolarProb_Moller'))
    #suite.addTest(TestScatterAnglesforDS('TestAzimProb_Moller'))
    #suite.addTest(TestScatterAnglesforDS('Test_crossSection'))
    suite.addTest(TestScatterAnglesforDS('TestPolarProb_Gryz_E0'))
    #suite.addTest(TestScatterAnglesforDS('TestPolarProb_Gryz'))
    #suite.addTest(TestScatterAnglesforDS('TestAzimProb_Gryz'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())