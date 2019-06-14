#!/usr/bin/env python

import numpy as np
import time
import sys
import getopt

from multiprocessing import Pool, cpu_count
from functools import partial

from scimath.units.energy import J, eV, KeV
from scimath.units.length import m, cm, km, angstrom
from scimath.units.api import UnitScalar, UnitArray

from material import material
from integrals import trapez_table, extF_limits_gryz, extF_limits_moller
from singleScatter import singleScatter_DS, singleScatter_cont
from extFunctions import gryz_dCS, moller_dCS
from parameters import c_pi_efour
from multiScatter import multiScatter_DS, multiScatter_cont
from fileTools import readInput, writeBSEtoHDF5

# if the script is run with the -u option
# run all the code with units and return units
# useful when in doubt about units
def main(argv):
    use_units = False
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv, "uhi:", ["ifile="])
    except getopt.GetoptError:
        print 'doScatter.py -i <input file> '
        print 'OR'
        print 'doScatter.py -u -i <input file>, if you want to track units'
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print 'doScatter.py -i <input file> '
            print 'OR'
            print 'doScatter.py -u -i <input file>, if you want to track units'
            sys.exit()
        elif opt == "-u":
            print "You chose to run scattering with units"
            use_units = True
        elif opt == "-i":
            inputfile = arg
            print "Input file is", arg

    return use_units, inputfile





if __name__ == '__main__': #this is necessary on Windows

    # use units?
    use_units, inputFile = main(sys.argv[1:])

    # read input parameters
    inputParameter = readInput(inputFile)

    # set material
    thisMaterial = material(inputParameter['material'])
    print 'scattering in', thisMaterial.species

    print 'scattering mode is:', inputParameter['mode']

    if use_units:
        # Set all input parameters with units, make calculations @with_units
        # and return values with units
        # With_units get toggled on if input is of unit type
        inputParameter.update({'E0': UnitScalar(inputParameter['E0'], units = 'eV')})
        inputParameter.update({'Emin': UnitScalar(inputParameter['Emin'], units = 'eV')})
        inputParameter.update({'Wc': UnitScalar(inputParameter['Wc'], units = 'eV')})

        # update material parameters to unit type
        thisMaterial.set_units()


    if (inptParams['mode'] == 'DS'):
         print '---- calculating Moller tables'
         tables_moller = trapez_table( inputParameter['E0'], inputParameter['Emin'],\
                                       np.array(inputParameter['Wc']), thisMaterial.fermi_e,\
                                       thisMaterial.params['n_val'], moller_dCS,\
                                       inputParameter['nBinsW'], inputParameter['nBinsE'] )

         print '---- calculating Gryzinski tables'
         tables_gryz = trapez_table( inputParameter['E0'], inputParameter['Emin'],\
                                     thisMaterial.params['Es'], thisMaterial.fermi_e,\
                                     thisMaterial.params['ns'], gryz_dCS,\
                                     inputParameter['nBinsW'], inputParameter['nBinsE'] )




    num_proc = cpu_count()-1 # leave one cpu thread free
    print 'You have', num_proc+1, "CPUs. I'm going to use", num_proc, 'of them'

    print '---- starting scattering'

    time_start = time.time()
    p = Pool(processes=num_proc)
    if (mode == 'DS'):
        f = partial(multiScatter_DS, material=material, E0=E0, Emin=Emin, tilt=tilt, tables_moller=tables_moller, tables_gryz=tables_gryz, Wc=Wc, units = units, parallel=True)
    else:
        f = partial(multiScatter_cont, material=material, E0=E0, Emin=Emin, tilt=tilt, units=units, parallel=True)

    # each worker gets num_electrons/num_proc
    BSE_data = p.map(f, [num_el/num_proc for _ in range(num_proc)])
    # BSE_data = p.map_async(f, xrange(num_el/num_proc))

    # serial
    #BSE_data = multiScatter_cont(num_el, material, E0, Emin, tilt, parallel = False)

    p.close()
    # if some things go wrong in the parallel code, pool.join() will throw some errosr
    p.join()

    # print progress
    # while(True):
    #     if (BSE_data.ready()):
    #         break
    #     remaining = BSE_data._number_left
    #     print 'Waiting for ', remaining, 'electrons to finish scattering'
    #     time.sleep(0.5)

    print 'time spent in scattering', time.time()-time_start
    print

    # save to file
    fileBSE = 'data/Al_BSE'+str(int(tilt))+'_tilt_'+mode+'_Emin'+str(int(Emin))+'_Emax'+str(int(E0))+'_bins'+nBinsE+'.h5'

    from parameters import alpha, xy_PC, L
    writeBSEtoHDF5(BSE_data, fileBSE, alpha, xy_PC, L)


    print 'BSE data had been written to ', fileBSE
