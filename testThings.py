import unittest

from multiprocessing import Process, cpu_count, Queue

from multiScatter import scatterMultiEl_cont, scatterMultiEl_DS, recover
from material import material


def pol2cart(rho, phi):
    ' From polar coordinates to '
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


# test the Monte Carlo results match the underlying probability ditribution from which events are sampled

class TestScatterAnglesforDS(unittest.TestCase):
    def __init__(self, inputPar, tables):
        self.inputPar = inputPar
        self.tables  = tables
        self.thingsToSave = {'scat_output': ['type', 'az_angle', 'polar_angle']}
        self.file = 'testData/testAngles_DS.temp'

    def SetUp(self):
        num_proc = cpu_count()-1
        output = {'scatterings': Queue()}

        processes = [Process(target=scatterMultiEl_DS, args=(self.inputPar, self.Tables,
                                            self.thingsToSave, output, count)) for count in range(num_proc)]
        # start threads
        for p in processes:
            p.start()

        # get results from queue
        results = recover(processes, output)

        # wait for processes to end
        for p in processes:
            # wait until all processes have finished
            p.join()
            p.terminate()

        # save to file
        writeBSEtoHDF5(results, inputPar, self.file)


    def TestAzimProb_Moller(self):
        '''
            Test if the azimutal angle distribution of Moller scatterings
            is within a certain range from the analytical probability distribution
        '''
        scatterings = pd.read_hdf(self.file, 'scatterings')
        calculatedAngles = scatterings[scatterings.type='Moller']['az_angle']
