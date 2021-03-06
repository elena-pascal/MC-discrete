#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys
from os import path, makedirs
import warnings
from getopt import GetoptError, getopt

import multiprocessing as mp
#import multiprocess as mp
from functools import partial
from tqdm import tqdm
#from scimath.units.api import UnitScalar, UnitArray
from memory_profiler import profile
from tools.material import material
from tools.probTables import genTables
from tools.multiScatter import multiTraj_DS,  multiTraj_cont, retrieve
from tools.fileTools import readInput, zipDict


class MapScatterer(object):
    ''' A map of the scatter_func on the number of electrons
    The listener function collects results from the Queue
    and appends them to the HDFstore
    '''
    def __init__(self, inputPar, scatter_func, listen_func, num_workers=mp.cpu_count()-1):
        '''
        inputPar
            Input parameters in dictionary form

        scatter_func
            Function to do the one electron scattering
            It must take only one paramer

        listen_func
            Function that listens to the queue

        num_workers
            The numbers of workers to create in the pool.
            Default is the number of CPU available - 1
        '''
        self.inputPar     = inputPar
        self.worker       = scatter_func
        self.listener     = listen_func
        self.pool         = mp.Pool(num_workers, maxtasksperchild=200)
        self.targetMat    = material(inputPar['material'])

    #@profile
    def error_call(self, result):
        print ('failed to callback result')

    #@profile
    def __call__(self, numJobs, storeName):
        '''
        numTrajPerJob
            number of trajectories per job

        storeName
            HDF5 store name
        '''
        # make a pandas store
        store = pd.HDFStore(storeName)

        # write input parameters to pandas series -> transposed dataframe
        store['input'] = pd.DataFrame(pd.Series(self.inputPar)).T
        store.close()

        # progress bar
        pbar = tqdm(total=self.inputPar['num_el'], desc='Trajectories finished:')

        # define the size of string columns in pandas hdf5 store
        strSize_dict = {'els':{}, 'scats':{}}

        if 'outcome' in self.inputPar['electron_output']:
            strSize_dict['els']['outcome'] = 6

        if 'type' in self.inputPar['scatter_output']:
            strSize_dict['scats']['type'] = 6

        # simplify the listener function to only depend on returned results from worker
        listenerWithStore = partial(self.listener, storeName=storeName,
                                                    strSize_dict=strSize_dict,
                                                    pbar=pbar)

        #logger.info('Starting multithreading')
        for _ in range(numJobs):
            foo=self.pool.apply_async(self.worker,
                            callback = listenerWithStore,
                            error_callback = self.error_call)

        # clean up
        self.pool.close()
        self.pool.join()
        print (' BSE data had been written to ', storeName)


#@profile
def listener(results, storeName, strSize_dict, pbar):
    '''
    Listens for results and appending them to the hdf5 store
    '''
    # results dictionary
    results_d = {'els':{}, 'scats':{}}

    # results arrives as a list of dictionaries,
    # make a simple dictionary with zipped lists
    for result in results:
        for key in results_d.keys():
            results_d[key] = zipDict(results_d[key], result[key])

    with pd.HDFStore(storeName) as store:
        for key in results_d.keys():
            # append the results to the store
            df = pd.DataFrame.from_dict(results_d[key])
            store.put(key, df, format='table', data_columns=True, append=True,
                        min_itemsize=strSize_dict[key])

    pbar.update(n=len(results))

def dealWithInput(argv):
    '''
    if the script is run with the -u option
    run all the code with units and return resulting units
    '''
    use_units = False
    inputfile = '../inputFile/input.file'

    def usage():
        print ('doScatter.py -i <input file> ')
        print ('              OR')
        print ('doScatter.py -u -i <input file>, if you want to track units \n')

    try:
        opts, _ = getopt(argv, "uhi:", ["ifile="])
    except GetoptError as err:
        print(err)
        usage()
        sys.exit()
    for opt, arg in opts:
        if opt == "-h":
            usage()
            sys.exit()
        elif opt == "-u":
            print ("\n You chose to run scattering with units \n")
            use_units = True
        elif opt == "-i":
            inputfile = arg
            print ("\n Input file given is: %s \n" %arg)

    return use_units, inputfile



def main():
    # read the comand line input if any
    use_unts, inputFile = dealWithInput(sys.argv[1:])

    # read input parameters
    inputPar = readInput(inputFile)

    # return some info about this run
    print(' \n - Number of incident electrons: %s' %inputPar['num_el'])
    print(' \n - Material is: %s' %inputPar['material'])
    print(' \n - Scattering mode is: %s ' %inputPar['mode'])
    print(' \n - Elastic scattering mode is: %s \n' % inputPar['elastic'])

    # generate integration tables instances
    my_tables = genTables(inputPar)

    # pandas is going to complain about performace for the input string table
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    # name the hdf file that stores the results
    storePath = '../data/'
    storeFile = storePath + 'TRSM_diff:'  + str(inputPar['diffMFP'])     +\
                            '_thick:' + str(inputPar['thickness'])  +\
                            '_mat:'   + str(inputPar['material'])   +\
                            '_mode:' + str(inputPar['mode'])       +\
                            '_elastic:' + str(inputPar['elastic']) +\
                            '_tilt:' + str(inputPar['s_tilt'])     +\
                            '_Emin:' + str(inputPar['Emin'])       +\
                            '_E0:'   + str(inputPar['E0'])         +\
                            '_tolE:' + str(inputPar['tol_E'])      +\
                            '_tolW:' + str(inputPar['tol_W'])      +\
                            '.h5'


    # check if storePath exists, if not create it
    if not path.exists(storePath):
        makedirs(storePath)

    # if file already exists delete it
    elif path.exists(storeFile):
        remove(storeFile)

    # make a new store
    store = pd.HDFStore(storeFile, mode='w')

    # save the input parameters to store
    store['input'] = pd.Series(inputPar.values(),
                               index=inputPar.keys(),
                               dtype=str)
    store.close()

    # set the material
    target_material = material(inputPar['material'])

    # make a dictionary with objects to save
    whatToSave = {'el_output': inputPar['electron_output'],
                 'scat_output': inputPar['scatter_output'] }


    # define number of traj per job
    #numTrajPerJob = 20
    numTrajPerJob = 4

    # define number of workers
    #num_workers = 11
    num_workers = 1

    # we need to spawn jobs = total number of electron trajctories/
    #                          num of trajectories per worker
    numJobs = int(inputPar['num_el']/numTrajPerJob)
    print ('There are %s jobs to be distributed across %s workers' %(numJobs, num_workers))

    if (inputPar['mode']=='DS'):
        # simplify worker function
        simplifiedWorker = partial(multiTraj_DS, inputPar = inputPar,
                                               numTraj = numTrajPerJob,
                                               material = target_material,
                                               tables = my_tables,
                                               thingsToSave = whatToSave)

        # instace of scatter mapper
        scatter = MapScatterer(inputPar, simplifiedWorker, listener, num_workers)

        # scatter object sent to multithreading
        scatter(numJobs, storeName=storeFile)

    else:
        # simplify worker function
        simplifiedWorker = partial(multiTraj_cont, inputPar = inputPar,
                                               numTraj = numTrajPerJob,
                                               material = target_material,
                                               tables = my_tables,
                                               thingsToSave = whatToSave)
        # instace of scatter mapper
        scatter = MapScatterer(inputPar, simplifiedWorker, listener, num_workers)

        # scatter object sent to multithreading
        scatter(numJobs, storeName=storeFile)




if __name__ == '__main__': #this is necessary on Windows

    main()
