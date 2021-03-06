import numpy as np
import os
import sys
import logging
import random
import time
import pickle
import cProfile
from tqdm import tqdm

from .electron import electron
from .singleScatter import trajectory_DS, trajectory_cont_cl
from .material import material

def scatterMultiEl_DS(inputPar, tables, listParToSave, outputQ, count):
    '''
    Does num electron scatterings. This can then be the target called by multiprocessing.
    '''
    # For parallel processes we need to make sure the random number seeds are different
    # Use, for instance, the process id multiplied by the current time
    random.seed(os.getpid()*int(time.time())) # getip only works on Unix

    # initialise new electron
    pos0 = np.array([0., 0., 0.,])
    dir0 = np.array([-np.sin(np.radians(inputPar['s_tilt'])),
                    0.,
                    np.cos(np.radians(inputPar['s_tilt']))])
    # patricks coordinates definition:
    #dir0 = np.array([np.cos(np.radians(90.0-tilt)), 0., -np.sin(np.radians(90.-tilt))])

    # make an iterator along the number of scattering events per process
    if (count == 0):
        # print progress bar for the first thread
        def iterator(num_el):
            return tqdm(range(num_el), desc='Scattering electrons')
    else:
        def iterator(num_el):
            return range(num_el)

    # define material
    targetMaterial = material(inputPar['material'])

    for _ in iterator(inputPar['num_el']):
        # start this electron
        el = electron(inputPar['E0'], inputPar['Emin'], pos0, dir0, listParToSave)

        # scatter a full trajectory
        trajectory_DS(el, targetMaterial, inputPar['Wc'], inputPar['maxScatt'], inputPar['elastic'], tables)

        try:
            # make tuples out of dictionaries and pickle them and them add to the queue
            #outputQ['els'].put(el.el_output.dict )
            #outputQ['scats'].put(el.scat_output.dict)
            outputQ['els']=el.el_output.dict
            outputQ['scats']=el.scat_output.dict
        except :
            print ( "Unexpected error when pickling results:", sys.exc_info()[0])
            raise

    # flush info to terminal
    sys.stdout.flush()




def multiTraj_DS(inputPar, numTraj, material, tables, thingsToSave):
    '''
    Does numTraj full trajectory scatterings. This can then be the target
    called by the multiprocessing pool.

    Input
        inputPar
            a dictionary of user defined parameters

        numTraj (per job)
            largest number of trajectories per process safe to keep in memory

        material
            an instance of the Material class defining the target material

        tables
            dictionary containing the integration tables

        thingsToSave
            dictionary containing the list of parameters to save
            This will be populated with the results

    Returns
        results
            a list of thingsToSave dictionaries populated with results
    '''
    # For parallel processes we need to make sure the random number seeds are different
    # Use, for instance, the process id multiplied by the current time
    random.seed(os.getpid()*int(time.time())) # getip only works on Unix

    # initialise new electron position and direction
    pos0 = np.array([0., 0., 0.,])
    dir0 = np.array([-np.sin(np.radians(inputPar['s_tilt'])),
                    0.,
                    np.cos(np.radians(inputPar['s_tilt']))])


    results = []

    for _ in range(numTraj):
        # start this electron
        el = electron(inputPar['E0'], inputPar['Emin'], pos0, dir0, thingsToSave)

        # scatter a full trajectory
        trajectory_DS(el, material, inputPar['Wc'], inputPar['maxScatt'], inputPar['thickness'], inputPar['elastic'], tables, inputPar['diffMFP'])

        results.append({'els':el.el_output.dict, 'scats':el.scat_output.dict})

        del el # not sure if this helps memory clean up in any way

    return results

def scatterMultiEl_cont(inputPar, thingsToSave, output, num, count):
    # for parallel processes we need to make sure the random number seeds are different
    # use for instance the process id multiplied by the current time
    random.seed(os.getpid()*int(time.time())) # getip only works on Unix

    pos0 = np.array([0., 0., 0.,])
    dir0 = np.array([-np.sin(np.radians(inputPar['s_tilt'])), 0., np.cos(np.radians(inputPar['s_tilt']))])

    # set the scattering function according to the choice of Bethe model
    if (inputPar['Bethe_model'] == 'classical'):
        def trajectory_cont(e_i, material, maxScatt):
            return trajectory_cont_cl(e_i, material, maxScatt)
# TODO:  JL and explicit  trajectories
    elif (Bethe_model == 'JL'):
        def scatterOneEl_cont(e_i, material, Emin):
            return scatterOneEl_cont_JL(e_i, material, Emin)

    elif (Bethe_model == 'explicit'):
        def scatterOneEl_cont(e_i, material, Emin):
            return scatterOneEl_cont_expl(e_i, material, Emin)

    else :
        print ('! I did not understand your choice of Bethe model in multiScatter')
        print ('! Exiting...')
        sys.exit()

    if (count == 0):
        # print progress bar for the first thread
        def iterator(num_el):
            return tqdm(range(num_el), desc='Scattering electrons')
    else:
        def iterator(num_el):
            return range(num_el)

    for _ in iterator(num):
        # start this electron
        el = electron(inputPar['E0'], inputPar['Emin'], pos0, dir0, thingsToSave)

        # profiler
        #cProfile.runctx('scatterOneEl_cont(e_i, material, Emin)', globals(), locals(), 'prof%d_cont.prof' %count)

        # scatter a full trajectory
        trajectory_cont(el, material(inputPar['material']), inputPar['maxScatt'], inputPar['thickness'])

    try:
        # make tuples out of dictionaries and pickle them
        output['electrons'].put(pickle.dumps(thingsToSave['el_output'].dict , protocol=2 ) )
        output['scatterings'].put(pickle.dumps(thingsToSave['scat_output'].dict , protocol=2 ) )
    except:
        print (" Unexpected error when pickling results:", sys.exc_info()[0])
        raise


def multiTraj_cont(inputPar, numTraj, material, thingsToSave, tables):
    '''
    Does numTraj full trajectory scatterings. This can then be the target
    called by the multiprocessing pool.

    Input
        inputPar
            a dictionary of user defined parameters

        numTraj (per job)
            largest number of trajectories per process safe to keep in memory

        material
            an instance of the Material class defining the target material

        thingsToSave
            dictionary containing the list of parameters to save
            This will be populated with the results

    Returns
        results
            a list of thingsToSave dictionaries populated with results
    '''
    # For parallel processes we need to make sure the random number seeds are different
    # Use, for instance, the process id multiplied by the current time
    random.seed(os.getpid()*int(time.time())) # getip only works on Unix

    # initialise new electron position and direction
    pos0 = np.array([0., 0., 0.,])
    dir0 = np.array([-np.sin(np.radians(inputPar['s_tilt'])),
                    0.,
                    np.cos(np.radians(inputPar['s_tilt']))])

    # set the scattering function according to the choice of Bethe model
    if (inputPar['Bethe_model'] == 'classical'):
        def trajectory_cont(e_i, material, maxScatt, elastic, tables):
            return trajectory_cont_cl(e_i, material, maxScatt, elastic, tables)

    elif (Bethe_model == 'JL'):
        def scatterOneEl_cont(e_i, material, Emin):
            return scatterOneEl_cont_JL(e_i, material, Emin)

    elif (Bethe_model == 'explicit'):
        def scatterOneEl_cont(e_i, material, Emin):
            return scatterOneEl_cont_expl(e_i, material, Emin)

    results = []
    for _ in range(numTraj):
        # start this electron
        el = electron(inputPar['E0'], inputPar['Emin'], pos0, dir0, thingsToSave)

        # scatter a full trajectory
        trajectory_cont(el, material, inputPar['maxScatt'], inputPar['elastic'], tables)

        results.append({'els':el.el_output.dict, 'scats':el.scat_output.dict})

        del el # not sure if this helps memory clean up in any way

    return results

def retrieve(jobs, output):
    ''' Retrieve results from Queue
        The processes are set up such that there are p results items per queue

        input :
            jobs   : list of processes threads
            output : {'electrons' : Queue(), 'scatterings' : Queue()}

        return:
            results: {'electrons' : list, 'scatterings' : list}
    '''

    results = {}
    for key in output.keys():
        results[key] = []

    for p in jobs:
        for key in output.keys():
            results[key].append(output[key].get())

    return results
