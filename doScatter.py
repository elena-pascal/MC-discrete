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
from rayScatter import rayScatter_cont
from extFunctions import gryz_dCS, moller_dCS
from parameters import c_pi_efour
from multiScatter import multiScatter_DS, multiScatter_cont
from writeHDF5 import writeBSEtoHDF5


if __name__ == '__main__': #this is necessary on Windows
    num_el = 100
    E0 = UnitScalar(10000, units = 'eV') # eV
    Emin = UnitScalar(9000, units = 'eV') # eV
    tilt = 70. # degrees
    mode = 'cont' # discrete scattering

    print 'scattering mode is:', mode
    nBinsW = 10
    nBinsE = 10

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




    num_proc = cpu_count()
    print 'you have', num_proc, "cpu's; not all of them physical"

    print '---- starting scattering'

    p = Pool(processes=num_proc-2)
    if (mode == 'DS'):
        f = partial(multiScatter_DS, material=material, E0=E0, Emin=Emin, tilt=tilt, tables_moller=tables_moller, tables_gryz=tables_gryz, Wc=Wc, parallel=True)
    else:
        f = partial(multiScatter_cont, material=material, E0=E0, Emin=Emin, tilt=tilt, parallel=True)

    # each worker gets num_electrons/num_proc
    BSE_data = p.map(f, [num_el/num_proc for _ in range(num_proc)])

    # serial
    #BSE_data_dict = multiScatter_cont(num_el, material, E0, Emin, tilt, parallel = False)

    p.close()
    # if some things go wrong in the parallel code, pool.join() will throw some errosr
    p.join()


    #print 'rate of BSE', len(BSE_data_dict['energy'])*1./num_el


    # save to file
    fileBSE = 'BSE'+str(int(tilt))+'tilt'+mode+'.h5'

    writeBSEtoHDF5(BSE_data, fileBSE)
