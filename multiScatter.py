import numpy as np
import os
import sys
import random
import pickle
from tqdm import tqdm
import cProfile

from material import material
from electron import electron, trajectory_DS

def scatterMultiEl_DS(inputPar, tables, thingsToSave, output, count):
    # for parallel processes we need to make sure the random number seeds are different
    # use for instance the process id multiplied by the current time

    # if parallel use a different random number seed per thread
    random.seed(os.getpid()) # getip only works on Unix

    # initialise new electron
    pos0 = np.array([0., 0., 0.,])
    dir0 = np.array([-np.sin(np.radians(inputPar['s_tilt'])), 0., np.cos(np.radians(inputPar['s_tilt']))])
    # patricks coordinates definition:
    #dir0 = np.array([np.cos(np.radians(90.0-tilt)), 0., -np.sin(np.radians(90.-tilt))])

    if (count == 0):
        # print progress bar for the first thread
        def iterator(num):
            return tqdm(range(num), desc='Scattering electrons')
    else:
        def iterator(num):
            return range(inputPar['num_el'])

    for _ in iterator(inputPar['num_el']):
        # start this electron
        el = electron(inputPar['E0'], inputPar['Emin'], pos0, dir0, thingsToSave)

        # profiler
        #cProfile.runctx('scatterOneEl_DS(e_i, material, Emin, Wc, table_moller, tables_gryz)', globals(), locals(), 'prof%d_ds.prof' %count)

        # scatter a full trajectory
        trajectory_DS(el, inputPar['E0'], material(inputPar['material']), inputPar['Wc'], inputPar['maxScatt'], tables)

    try:
        # make tuples out of dictionaries and pickle them
        output['electrons'].put(pickle.dumps(thingsToSave['el_output'].dict , protocol=2 ) )
        output['scatterings'].put(pickle.dumps(thingsToSave['scat_output'].dict , protocol=2 ) )
    except :
        print ( "Unexpected error when pickling results:", sys.exc_info()[0])
        raise



def scatterMultiEl_cont(num_el, material, E0, Emin, tilt, Bethe_model, thingsToSave, output, count):
    # for parallel processes we need to make sure the random number seeds are different
    # use for instance the process id multiplied by the current time
    #if parallel:
    random.seed(os.getpid()) # getip only on Unix

    pos0 = np.array([0., 0., 0.,])
    dir0 = np.array([-np.sin(np.radians(tilt)), 0., np.cos(np.radians(tilt))])

    # set the scattering function according to the choice of Bethe model
    if (Bethe_model == 'classical'):
        def scatterOneEl_cont(e_i, material, Emin):
            return scatterOneEl_cont_cl(e_i, material, Emin)

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

    indexes = ('outcome', 'energy', 'dir', 'MFP', 'TP', 'num_scatt')
    out_dict = {}

    for label in indexes:
        out_dict[label] = []

    if (count == 0):
        # print progress bar for the first thread
        def iterator(num):
            return tqdm(range(num), desc='number of scattered electrons per thread')
    else:
        def iterator(num):
            return range(num_el)

    for _ in iterator(num_el):
        # start this electron
        e_i = electron(E0, Emin, pos0, dir0, thingsToSave)

        # profiler
        #cProfile.runctx('scatterOneEl_cont(e_i, material, Emin)', globals(), locals(), 'prof%d_cont.prof' %count)

        # scatter until end of scatter
        res_dict = scatterOneEl_cont(e_i, material, Emin)

        # append data for all electrons
        for index in res_dict.keys():
            out_dict[index].append(res_dict[index])

        out_dict['outcome'].append(e_i.outcome)
        out_dict['energy'].append(e_i.energy)
        out_dict['dir'].append(tuple(e_i.dir))

    # TODO: tuple instead of dict ?
    try:
        # make tuples out of lists and pickle them
        output.put(pickle.dumps( (indexes,  tuple( tuple(out_dict[label]) for label in indexes)) , protocol=2 ) )

    except:
        print (" Unexpected error:", sys.exc_info()[0])
        raise



def recover(jobs, output):
    ''' Recover results from Queue
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
            results[key].append(pickle.loads(output[key].get()))

    return results
