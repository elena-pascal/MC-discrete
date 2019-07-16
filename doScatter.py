#!/usr/bin/env python3

import numpy as np
import time
import sys
import getopt
import pickle

from multiprocessing import Process, Queue, cpu_count
from functools import partial
from scimath.units.api import UnitScalar, UnitArray
from tqdm import tqdm

from material import material
from integrals import trapez_table, extF_limits_gryz, extF_limits_moller
from extFunctions import gryz_dCS_P, moller_dCS
from parameters import u_pi_efour

from electron import electron
from singleScatter import scatterOneEl_DS_wUnits, scatterOneEl_cont_cl_wUnits, scatterOneEl_cont_JL_wUnits, scatterOneEl_cont_expl_wUnits
from multiScatter import scatterMultiEl_DS, scatterMultiEl_cont
from fileTools import readInput, writeAllEtoHDF5

# if the script is run with the -u option
# run all the code with units and return units
# useful when in doubt about units
def main(argv):
    use_units = False
    inputfile = 'input.file'

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

    # use units?
    use_units, inputFile = main(sys.argv[1:])

    # read input parameters
    inputPar = readInput(inputFile)

    # set material
    thisMaterial = material(inputPar['material'])

    # number of processes available
    num_proc = cpu_count()-1 # leave one cpu thread free for user
    #num_proc = 1

    print()
    print(' number of incident electrons:', inputPar['num_el']*num_proc)
    print()
    print(' material is:', thisMaterial.species)
    print()
    print(' scattering mode is:', inputPar['mode'])
    print()

    if (inputPar['mode'] == 'DS'):
         print ('---- calculating Moller tables')
         tables_moller = trapez_table( inputPar['E0'], inputPar['Emin'],\
                                       np.array([inputPar['Wc']]), thisMaterial.fermi_e,\
                                       np.array([thisMaterial.params['n_val']]), moller_dCS,\
                                       inputPar['num_BinsW'], inputPar['num_BinsE'] )

         print ('---- calculating Gryzinski tables')
         tables_gryz = trapez_table( inputPar['E0'], inputPar['Emin'],\
                                    thisMaterial.params['Es'], thisMaterial.fermi_e,\
                                    thisMaterial.params['ns'], gryz_dCS_P,\
                                    inputPar['num_BinsW'], inputPar['num_BinsE'] )

    # elif (inputPar['mode'] in ['diel', 'dielectric']):
    #     print ' ---- calculating dielectric function integral table'
    #     tables_diel =

############################## units###########################################
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
            if (inputPar['Bethe'] == 'classical'):
                scatterOneEl_cont_cl_wUnits(oneElectron, thisMaterial, inputPar['Emin'])
            elif (inputPar['Bethe'] == 'JL'):
                scatterOneEl_cont_JL_wUnits(oneElectron, thisMaterial, inputPar['Emin'])
            elif (inputPar['Bethe'] == 'explicit'):
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





    print()
    print (' you have', num_proc+1, "CPUs. I'm going to use", num_proc, 'of them')
    print()

    output = Queue()

    # define the function for scattering of multiple electrons depending on the model
    if (inputPar['mode'] == 'DS'):
        processes = [Process(target=scatterMultiEl_DS, args=(inputPar['num_el'], thisMaterial,
                                                            inputPar['E0'], inputPar['Emin'],
                                                            inputPar['s_tilt'], tables_moller,
                                                            tables_gryz, inputPar['Wc'],
                                                            output, count)) for count in range(num_proc)]

    elif (inputPar['mode'] == 'cont'):
        processes = [Process(target=scatterMultiEl_cont, args=(inputPar['num_el'], thisMaterial,
                                                              inputPar['E0'], inputPar['Emin'],
                                                              inputPar['s_tilt'], inputPar['Bethe'],
                                                              output, count)) for count in range(num_proc)]


    print ('---- starting scattering')
    time_start = time.time()

    # start threads
    for p in processes:
        p.start()

    # wait for processes to end # this deadlocks the pipe so moved it after queue.get
    # for p in processes:
    #    p.join()
    result = [pickle.loads(output.get()) for p in processes]

    print()
    print ('joining results ...')
    print()

    for p in processes:
        p.join()
        p.terminate()

    print ('---- finished scattering')

    print()
    print (' time spent in scattering', time.time()-time_start)
    print()

    # save to file
    fileBSE = 'data/Al_BSE_' + str(inputPar['mode'])+ '_normal_all.h5'


    print ('---- writting to file')
    time_start = time.time()

    from parameters import alpha, xy_PC, L
    writeAllEtoHDF5(result, inputPar, fileBSE, alpha, xy_PC, L)
    print()
    print (' time spent writting to file:', time.time()-time_start)
    print()
    print (' BSE data had been written to ', fileBSE)
    print()
