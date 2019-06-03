import numpy as np
import time

from multiprocessing import Pool, cpu_count
from functools import partial

from scimath.units.energy import J, eV, KeV
from scimath.units.length import m, cm, km, angstrom
from scimath.units.api import UnitScalar, UnitArray, convert, has_units

from material import material
from integrals import trapez_table, extF_limits_gryz, extF_limits_moller
from singleScatter import singleScatter_DS, singleScatter_cont
from extFunctions import gryz_dCS, moller_dCS
from parameters import c_pi_efour
from multiScatter import multiScatter_DS, multiScatter_cont
from writeHDF5 import writeBSEtoHDF5


if __name__ == '__main__': #this is necessary on Windows
    num_el = 100000
    E0 = UnitScalar(30000, units = 'eV') # eV
    Emin = UnitScalar(25000, units = 'eV') # eV
    tilt = 70. # degrees
    mode = 'DS' # discrete scattering

    print 'scattering mode is:', mode
    nBinsW = 10000
    nBinsE = 10000

    Wc = UnitScalar(5, units = 'eV')
    material = material('Al')


    def u2n(value_with_units):
        '''
        Tranforms quantity with units in
        numpy unitless array
        '''
        return np.array(value_with_units)

    if (mode == 'DS'):
        print '---- calculating Moller tables'
        # remove the function dependece on the constant and get rid of units
        funcToint_M = lambda E, W, n_e : u2n(moller_dCS(E, W, n_e, c_pi_efour))

        tables_moller = trapez_table( float(E0), float(Emin), np.array([Wc]), float(material.fermi_e()), \
                 np.array([material.nval()]), funcToint_M, nBinsW, nBinsE)

        print '---- calculating Gryzinski tables'
        tables_gryz = []

        funcToint_G = lambda E, W, n_e, Ebi : u2n(gryz_dCS(E, W, n_e,\
                                                 c_pi_efour, Ebi))

        tables_gryz = trapez_table(float(E0), float(Emin), np.array(material.Es()), float(material.fermi_e()),\
                     np.array(material.ns()), funcToint_G, nBinsW, nBinsE )




    num_proc = cpu_count()-1 # leave one cpu thread free
    print 'You have', num_proc+1, "CPUs"

    print '---- starting scattering'

    time0 = time.time()
    p = Pool(processes=num_proc)
    if (mode == 'DS'):
        f = partial(multiScatter_DS, material=material, E0=E0, Emin=Emin, tilt=tilt, tables_moller=tables_moller, tables_gryz=tables_gryz, Wc=Wc, parallel=True)
    else:
        f = partial(multiScatter_cont, material=material, E0=E0, Emin=Emin, tilt=tilt, parallel=True)

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

    print 'time spent in scattering', time.time()-time0
    print

    # save to file
    fileBSE = 'data/Al_BSE'+str(int(tilt))+'_tilt_'+mode+'_Emin'+str(int(Emin))+'_Emax'+str(int(E0))+'_bins'+nBinsE+'.h5'

    writeBSEtoHDF5(BSE_data, fileBSE)

    print 'BSE data had been written to ', fileBSE
