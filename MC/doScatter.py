#!/usr/bin/env python3

import numpy as np
import pandas as pd
import time
import sys
import logging
import getopt
#import pickle
import dill as pickle
#from multiprocessing import Process, cpu_count, Queue, log_to_stderr, get_logger
#from multiprocessing import Pool
import pathos.multiprocessing as multiprocessing
from pathos.helpers import mp as pathos_multiprocess
from functools import partial
from scimath.units.api import UnitScalar, UnitArray
from tqdm import tqdm

from MC.material import material
from MC.electron import electron
from MC.probTables import genTables
from MC.singleScatter import scatterOneEl_DS_wUnits, scatterOneEl_cont_cl_wUnits, scatterOneEl_cont_JL_wUnits, scatterOneEl_cont_expl_wUnits
from MC.singleScatter import trajectory_DS, trajectory_cont_cl
from MC.multiScatter import scatterMultiEl_DS, scatterMultiEl_cont, retrieve
from MC.fileTools import readInput, writeBSEtoHDF5, thingsToSave


class MapScatterer(object):
    ''' A map of the scatter_func on the number of electrons
    The listener function collects results from the Queue
    and appends them to the HDFstore
    '''
    def __init__(self, inputPar, scatter_func, listen_func, num_workers=None):
        '''
        inputPar
            Input parameters in dictionary form

        scatter_func
            Function to do the one electron scattering
            It must take only one paramer

        listen_func
            Function that listens to the queue

        thingsToSave
            A dictionary of data to save

        num_workers
            The numbers of workers to create in the pool.
            Default is the number of CPU available - 1
        '''
        self.inputPar     = inputPar
        self.worker       = scatter_func
        self.listener     = listen_func
        self.pool         = multiprocessing.Pool(num_workers)

        # set queue from manager
        self.q            = {'electrons' : pathos_multiprocess.Manager().Queue(),
                            'scatterings' : pathos_multiprocess.Manager().Queue()}

    def __call__(self, electrons, material, tables, thingsToSave, store):
        '''
        electrons
            An interable over the number of electrons

        chunksize=1
            The portion of total num_el to hand to each worker
        '''

        # start listener
        watcher = self.pool.apply_async(self.listener, (store, self.q))
        print ('---------------defined watcher')

        # fire off workers for every electron
        jobs = []

        for el in electrons:
            job = self.pool.apply_async(self.worker, (el, self.inputPar, material, tables, thingsToSave, self.q))
            jobs.append(job)
        print ('--------------fired off worker')

        # collect results from workers through the pool result quue
        for job in jobs:
            job.get()
        print ('-----------------collected jobs')
        # kill the listener when done
        self.q['electrons'].put('kill')
        self.q['scatterings'].put('kill')
        self.pool.close()
        self.pool.join()




# if the script is run with the -u option
# run all the code with units and return units
# useful when in doubt about units
def main(argv):
    use_units = False
    inputfile = '../inputFile/input.file'

    def usage():
        print ('doScatter.py -i <input file> ')
        print ('              OR')
        print ('doScatter.py -u -i <input file>, if you want to track units')
        print ()

    try:
        opts, _ = getopt.getopt(argv, "uhi:", ["ifile="])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit()
    for opt, arg in opts:
        if opt == "-h":
            usage()
            sys.exit()
        elif opt == "-u":
            print()
            print ("You chose to run scattering with units")
            print()
            use_units = True
        elif opt == "-i":
            inputfile = arg
            print ("Input file given is", arg)
            print()

    return use_units, inputfile





if __name__ == '__main__': #this is necessary on Windows

    # log information
    #multiprocessing.log_to_stderr()
    #logger = multiprocessing.get_logger()
    #logger.setLevel(logging.INFO)

    # use units?
    use_units, inputFile = main(sys.argv[1:])

    # read input parameters
    inputPar = readInput(inputFile)


    print()
    print(' number of incident electrons per processor:', inputPar['num_el'])
    print()
    print(' material is:', inputPar['material'])
    print()
    print(' scattering mode is:', inputPar['mode'])
    print()

    # compute integration tables
    if (inputPar['mode'] == 'DS'):
        # generate integration tables instances
        my_tables = genTables(inputPar)


############################## units ###########################################
    if use_units:
        # Set all input parameters with units, make calculations @with_units
        # and return pathlength and energy units
        # With_units gets toggled on if input is of unit type
        inputPar.update({'E0': UnitScalar(inputPar['E0'], units = 'eV')})
        inputPar.update({'Emin': UnitScalar(inputPar['Emin'], units = 'eV')})
        inputPar.update({'Wc': UnitScalar(inputPar['Wc'], units = 'eV')})

        # update material parameters to unit type
        thisMaterial.set_units()

        # scatter one electron
        pos0 = UnitArray((0., 0., 0.), units='angstrom')
        dir0 = (-np.sin(np.radians(inputPar['s_tilt'])), 0., np.cos(np.radians(inputPar['s_tilt'])))
        oneElectron = electron(inputPar['E0'], pos0, dir0)

        if (inputPar['mode'] == 'DS'):
            scatterOneEl_DS_wUnits(oneElectron, thisMaterial, inputPar['Emin'], inputPar['Wc'],  tables_moller, tables_gryz)
            print ('- Energy units:', oneElectron.energy.units)
            print ('- Distance units:', oneElectron.xyz.units)
            sys.exit()

        elif (inputPar['mode'] == 'cont'):
            if (inputPar['Bethe_model'] == 'classical'):
                scatterOneEl_cont_cl_wUnits(oneElectron, thisMaterial, inputPar['Emin'])
            elif (inputPar['Bethe_model'] == 'JL'):
                scatterOneEl_cont_JL_wUnits(oneElectron, thisMaterial, inputPar['Emin'])
            elif (inputPar['Bethe_model'] == 'explicit'):
                scatterOneEl_cont_expl_wUnits(oneElectron, thisMaterial, inputPar['Emin'])
            else:
                print (' ! I did not understand the Bethe model type in units check')
                print (' ! Exiting')
                sys.exit()

            print()
            print ('- Energy units:', oneElectron.energy.units)
            print ('- Distance units:', oneElectron.xyz.units)
            sys.exit()
        else:
            print()
            print (' I did not understand the input scattering mode')

###############################################################################

    def worker_DS(el_id, input, material, tables, thingsToSave, queue):
        '''
        Scatter one electron using the direct model
        and add the results to the queue
        '''
        # initialise new electron
        pos0 = np.array([0., 0., 0.,])
        dir0 = np.array([-np.sin(np.radians(input['s_tilt'])),
                        0., np.cos(np.radians(input['s_tilt']))])

        # make an electron instance
        el = electron(input['E0'], input['Emin'], pos0, dir0, thingsToSave)
        print ('----------------made electron')
        # scatter this electron a full trajectory
        trajectory_DS(el, material, input['Wc'], input['maxScatt'], tables)
        print  ('-----------did one trajectory')
        # add results to the queue
        try:
            # make tuples out of dictionaries and pickle them and them add to the queue
            queue['electrons'].put(pickle.dumps(thingsToSave['el_output'].dict , protocol=2 ) )
            queue['scatterings'].put(pickle.dumps(thingsToSave['scat_output'].dict , protocol=2 ) )
        except :
            print ( "Unexpected error when pickling results:", sys.exc_info()[0])
            raise

            # flush info to terminal
            #sys.stdout.flush()


    # name the hdf file that stores the results
    storeFile = '../data/Al_BSE' +   '_mode:' + str(inputPar['mode'])  +\
                                '_tilt:' + str(inputPar['s_tilt'])+\
                                '_Emin:' + str(inputPar['Emin'])  +\
                                '_E0:'   + str(inputPar['E0'])    +\
                                '_tolE:' + str(inputPar['tol_E']) +\
                                '_tolW:' + str(inputPar['tol_W']) +\
                                '.h5'

    print ('-----------------------instance done')

    def listener(store, queue):
        '''
        Listens for results in queue stores them
        '''

        while True:
            # get the results from queue
            results = queue.get()
            print ('results:', results)
            # stop listening when kill signal arrives
            if results == 'kill':
                break

            # append the results to the store
            writeBSEtoHDF5(results, store)


    # create HDF5 store
    store = pd.HDFStore(storeFile)

    # write input parameters to pandas series -> transposed dataframe
    store['input'] = pd.DataFrame(pd.Series(inputPar)).T

    target_material = material(inputPar['material'])

    whatToSave = {'el_output': thingsToSave(inputPar['electron_output']),
                 'scat_output': thingsToSave(inputPar['scatter_output']) }

    scatter = MapScatterer(inputPar, worker_DS, listener, 3)
    print ('\n instance done------')
    scatter([el for el in range(10)], target_material, my_tables, whatToSave, store)

    print (' BSE data had been written to ', storeFile)

    # # number of processes available
    # num_proc = cpu_count()-1 # leave one cpu thread free for user
    # #num_proc = 1
    #
    # print()
    # print (' you have', num_proc+1, "CPUs. I'm going to use", num_proc, 'of them')
    # print()
    #
    # output = {'electrons' : Queue(), 'scatterings' : Queue()}
    #
    # # dictionary of output objects
    # whatToSave = {'el_output': thingsToSave(inputPar['electron_output']),
    #             'scat_output': thingsToSave(inputPar['scatter_output']) }
    #
    # num = 10000
    # # define the function for scattering of multiple electrons depending on the model
    # if (inputPar['mode'] == 'DS'):
    #     processes = [Process(target=scatterMultiEl_DS, args=(inputPar, tables,
    #                                 whatToSave, output, num, count)) for count in range(int(inputPar['num_el']/num))]
    #
    # elif (inputPar['mode'] == 'cont'):
    #     processes = [Process(target=scatterMultiEl_cont, args=(inputPar,
    #                                 whatToSave, output, num, count)) for count in range(num_proc)]
    #
    #
    # print ('---- starting scattering')
    # time_start = time.time()
    #
    # # start threads
    # for p in processes:
    #     p.start()
    #
    # # get results from queue
    # results = retrieve(processes, output)
    #
    # print()
    # print ('joining results ...')
    # print()
    #
    # # wait for processes to end
    # for p in processes:
    #     # wait until all processes have finished
    #     p.join()
    #     p.terminate()
    #
    # print ('---- finished scattering')
    # print()
    # print (' time spent in scattering', time.time()-time_start)
    # print()
    #
    #
    # print ('---- writting to file')
    # time_start = time.time()
    #
    # writeBSEtoHDF5(results, inputPar, fileBSE)
    # print()
    # print (' time spent writting to file:', time.time()-time_start)
