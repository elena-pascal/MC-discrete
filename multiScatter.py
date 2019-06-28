import numpy as np
import os
import sys

from electron import electron
from singleScatter import scatterOneEl_DS, scatterOneEl_cont_cl, scatterOneEl_cont_JL, scatterOneEl_cont_expl


def scatterMultiEl_DS(num_el, material, E0, Emin, tilt, tables_moller, tables_gryz, Wc, parallel=False):
    # for parallel processes we need to make sure the random number seeds are different
    # use for instance the process id as seed
    if parallel:
        np.random.seed(os.getpid()) # only on Unix

    pos0 = np.array([0., 0., 0.,])
    dir0 = np.array([-np.sin(np.radians(tilt)), 0., np.cos(np.radians(tilt))])
    # patricks coordinates definition:
    #dir0 = np.array([np.cos(np.radians(90.0-tilt)), 0., -np.sin(np.radians(90.-tilt))])

    BSE_energy       = []
    BSE_dir          = []
    BSE_scatterAngle = []
    mean_pathl       = []
    num_scatt        = []

    for _ in range(num_el):
        # start this electron
        e_i = electron(E0, pos0, dir0)

        # scatter until end of scatter
        res_dict = scatterOneEl_DS(e_i, material, Emin, Wc, tables_moller, tables_gryz)
        mean_pathl.append(res_dict['mean_pathl'])
        num_scatt.append(res_dict['num_scattering'])

        # append data for the backscattered electrons
        if (e_i.outcome == 'backscattered'):
            BSE_energy.append(float(e_i.energy))
            BSE_dir.append(e_i.dir)

    # dictionary of form { 'BSE': {label: list}, 'all': {label:list}}
    return {'BSE':   {'energy' : BSE_energy, 'direction' : BSE_dir},
            'all':   {'mean_pathl' : mean_pathl, 'num_scatter' : num_scatt} }


def scatterMultiEl_cont(num_el, material, E0, Emin, tilt, Bethe_model, parallel=False):
    # for parallel processes we need to make sure the random number seeds are different
    # use for instance the process id a seed
    if parallel:
        np.random.seed(os.getpid()) # only on Unix

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
        print '! I did not understand your choice of Bethe model in multiScatter'
        print '! Exiting...'
        sys.exit()

    BSE_energy       = []
    BSE_dir          = []
    BSE_scatterAngle = []
    mean_pathl       = []
    num_scatt        = []

    for _ in range(num_el):
        # start this electron
        e_i = electron(E0, pos0, dir0)

        # scatter until end of scatter
        res_dict = scatterOneEl_cont(e_i, material, Emin)
        mean_pathl.append(res_dict['mean_pathl'])
        num_scatt.append(res_dict['num_scattering'])

        # append data for the backscattered electrons
        if (e_i.outcome == 'backscattered'):
            BSE_energy.append(e_i.energy)
            BSE_dir.append(e_i.dir)

    # dictionary of form { 'BSE': {label: list}, 'all': {label:list}}
    return {'BSE':   {'energy' : BSE_energy, 'direction' : BSE_dir},
            'all':   {'mean_pathl' : mean_pathl, 'num_scatter' : num_scatt} }
