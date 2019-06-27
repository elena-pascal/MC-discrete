#!/usr/bin/env python

import numpy as np
import time
import sys
import getopt

from multiprocessing import Pool, cpu_count
from functools import partial
from scimath.units.api import UnitScalar, UnitArray


from material import material
from integrals import trapez_table, extF_limits_gryz, extF_limits_moller
from extFunctions import gryz_dCS, moller_dCS
from extFunctions import gryz_dCS, moller_dCS
from parameters import u_pi_efour

from electron import electron
from singleScatter import scatterOneEl_DS_wUnits, scatterOneEl_cont_cl_wUnits, scatterOneEl_cont_JL_wUnits, scatterOneEl_cont_expl_wUnits
from multiScatter import scatterMultiEl_DS, scatterMultiEl_cont
from fileTools import readInput, writeBSEtoHDF5

# if the script is run with the -u option
# run all the code with units and return units
# useful when in doubt about units
def main(argv):
    use_units = False
    inputfile = 'input.file'

    def usage():
        print 'doScatter.py -i <input file> '
        print '              OR'
        print 'doScatter.py -u -i <input file>, if you want to track units'
        print

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
            print
            print "You chose to run scattering with units"
            print
            use_units = True
        elif opt == "-i":
            inputfile = arg
            print "Input file given is", arg
            print

    return use_units, inputfile





if __name__ == '__main__': #this is necessary on Windows

    # use units?
    use_units, inputFile = main(sys.argv[1:])

    # read input parameters
    inputParameter = readInput(inputFile)

    # set material
    thisMaterial = material(inputParameter['material'])
    print ' material is:', thisMaterial.species

    print ' scattering mode is:', inputParameter['mode']
    print

    if (inputParameter['mode'] == 'DS'):
         print '---- calculating Moller tables'
         tables_moller = trapez_table( inputParameter['E0'], inputParameter['Emin'],\
                                       np.array([inputParameter['Wc']]), thisMaterial.fermi_e,\
                                       np.array([thisMaterial.params['n_val']]), moller_dCS,\
                                       inputParameter['num_BinsW'], inputParameter['num_BinsE'] )

         print '---- calculating Gryzinski tables'
         tables_gryz = trapez_table( inputParameter['E0'], inputParameter['Emin'],\
                                     thisMaterial.params['Es'], thisMaterial.fermi_e,\
                                     thisMaterial.params['ns'], gryz_dCS,\
                                     inputParameter['num_BinsW'], inputParameter['num_BinsE'] )

    # elif (inputParameter['mode'] in ['diel', 'dielectric']):
    #     print ' ---- calculating dielectric function integral table'
    #     tables_diel =


    if use_units:
        # Set all input parameters with units, make calculations @with_units
        # and return pathlength and energy units
        # With_units gets toggled on if input is of unit type
        inputParameter.update({'E0': UnitScalar(inputParameter['E0'], units = 'eV')})
        inputParameter.update({'Emin': UnitScalar(inputParameter['Emin'], units = 'eV')})
        inputParameter.update({'Wc': UnitScalar(inputParameter['Wc'], units = 'eV')})

        # update material parameters to unit type
        thisMaterial.set_units()

        # scatter one electron
        pos0 = UnitArray(np.array([0., 0., 0.]), units='angstrom')
        dir0 = np.array([-np.sin(np.radians(inputParameter['s_tilt'])), 0., np.cos(np.radians(inputParameter['s_tilt']))])
        oneElectron = electron(inputParameter['E0'], pos0, dir0)

        if (inputParameter['mode'] == 'DS'):
            scatterOneEl_DS_wUnits(oneElectron, thisMaterial, inputParameter['Emin'], inputParameter['Wc'],  tables_moller, tables_gryz)
            print '- Energy units:', oneElectron.energy.units
            print '- Distance units:', oneElectron.xyz.units
            sys.exit()

        elif (inputParameter['mode'] == 'cont'):
            if (inputParameter['Bethe'] == 'classical'):
                scatterOneEl_cont_cl_wUnits(oneElectron, thisMaterial, inputParameter['Emin'])
            elif (inputParameter['Bethe'] == 'JL'):
                scatterOneEl_cont_JL_wUnits(oneElectron, thisMaterial, inputParameter['Emin'])
            elif (inputParameter['Bethe'] == 'explicit'):
                scatterOneEl_cont_expl_wUnits(oneElectron, thisMaterial, inputParameter['Emin'])
            else:
                print ' ! I did not understand the Bethe model type in units check'
                print ' ! Exiting'
                sys.exit()

            print
            print '- Energy units:', oneElectron.energy.units
            print '- Distance units:', oneElectron.xyz.units
            sys.exit()
        else:
            print
            print ' I did not understand the input scattering mode'



    num_proc = cpu_count()-1 # leave one cpu thread free
    print ' You have', num_proc+1, "CPUs. I'm going to use", num_proc, 'of them'
    print
    print '---- starting scattering'

    time_start = time.time()
    p = Pool(processes=num_proc)

    if (inputParameter['mode'] == 'DS'):
        f = partial(scatterMultiEl_DS, material=thisMaterial, E0=inputParameter['E0'],\
                    Emin=inputParameter['Emin'], tilt=inputParameter['s_tilt'],\
                    tables_moller=tables_moller, tables_gryz=tables_gryz,\
                    Wc=inputParameter['Wc'], parallel=True)
    elif (inputParameter['mode'] == 'cont'):
        f = partial(scatterMultiEl_cont, material=thisMaterial, E0=inputParameter['E0'],\
                    Emin=inputParameter['Emin'], tilt=inputParameter['s_tilt'], \
                    Bethe_model = inputParameter['Bethe'], parallel=True)


    # each worker gets num_electrons/num_proc
    BSE_data = p.map(f, [inputParameter['num_el']/num_proc for _ in range(num_proc)])
    # BSE_data = p.map_async(f, xrange(num_el/num_proc))
    print '---- finished scattering'
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

    print ' time spent in scattering', time.time()-time_start
    print

    # save to file
    fileBSE = 'data/Al_BSE_'+'tilt'+ str(int(inputParameter['s_tilt']))+ '_'+\
                   inputParameter['mode']+inputParameter['Bethe']+'_Emin'+str(int(inputParameter['Emin']))+\
                   '_Emax'+str(int(inputParameter['E0']))+'_bins'+str(inputParameter['num_BinsE'])+'.h5'

    from parameters import alpha, xy_PC, L
    writeBSEtoHDF5(BSE_data, fileBSE, alpha, xy_PC, L)


    print 'BSE data had been written to ', fileBSE
