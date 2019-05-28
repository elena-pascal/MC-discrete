from multiprocessing import Pool, Process, Queue
import numpy as np
import os


from electron import electron
from singleScatter import singleScatter_DS, singleScatter_cont



def multiScatter_DS(num_el, material, E0, Emin, tilt, tables_moller, tables_gryz, Wc, parallel=False):
    # for parallel processes we need to make sure the random number seeds are different
    # for instance the process id
    if parallel:
        np.random.seed(os.getpid()) # only on Unix

    pos0 = np.array([0., 0., 0.,])
    dir0 = np.array([-np.sin(np.radians(tilt)), 0., np.cos(np.radians(tilt))])
    # patricks coordinates definition:
    #dir0 = np.array([np.cos(np.radians(90.0-tilt)), 0., -np.sin(np.radians(90.-tilt))])

    BSE_energy = []
    BSE_dir = []
    BSE_scatterAngle = []

    for _ in range(num_el):
        # start this electron
        e_i = electron(E0, pos0, dir0)

        # scatter until end of scatter
        singleScatter_DS(e_i, material, Emin, Wc, tables_moller, tables_gryz)

        # append data for the backscattered electrons
        if (e_i.outcome == 'backscattered'):
            BSE_energy.append(float(e_i.energy))
            BSE_dir.append(e_i.dir)

    return [BSE_energy, BSE_dir]


def multiScatter_cont(num_el, material, E0, Emin, tilt, parallel=False):
    # for parallel processes we need to make sure the random number seeds are different
    # for instance the process id
    if parallel:
        np.random.seed(os.getpid()) # only on Unix

    pos0 = np.array([0., 0., 0.,])
    dir0 = np.array([-np.sin(np.radians(tilt)), 0., np.cos(np.radians(tilt))])

    BSE_energy = []
    BSE_dir = []
    BSE_scatterAngle = []

    for _ in range(num_el):
        # start this electron
        e_i = electron(E0, pos0, dir0)

        # scatter until end of scatter
        singleScatter_cont(e_i, material, Emin)

        # append data for the backscattered electrons
        if (e_i.outcome == 'backscattered'):
            BSE_energy.append(float(e_i.energy))
            BSE_dir.append(e_i.dir)

    return {'energy':BSE_energy, 'direction':BSE_dir}
