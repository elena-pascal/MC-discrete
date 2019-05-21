import ray
import numpy as np


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


import time
from multiScatter import multiScatter_DS

num_el = 1000
E0 = UnitScalar(30000, units = 'eV') # eV
Emin = UnitScalar(15000, units = 'eV') # eV
tilt = 70. # degrees
mode = 'DS' # discrete scattering

nBinsW = 1000
nBinsE = 1000

Wc = UnitScalar(5, units = 'eV')
material = material('Al')


def u2n(value_with_units):
    '''
    Tranforms quantity with units in
    numpy unitless array
    '''
    return np.array(value_with_units)

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



if __name__ == '__main__': #this is necessary on Windows
    num_proc = cpu_count()
    print 'you have', num_proc, "cpus"

    t1 = time.time()
    p = Pool(processes=num_proc-2)

    f = partial(multiScatter_DS, material=material, E0=E0, Emin=Emin, tilt=tilt, tables_moller=tables_moller, tables_gryz=tables_gryz, Wc=Wc, parallel=True)
    # each worker gets num_electrons/num_proc
    (BSEs, pathlen, thetas) = p.map(f, [num_el/num_proc for _ in range(num_proc)])
    p.close()
    # if some things go wrong in the parallel code, pool.join() will throw some errosr
    p.join()

    print 'pool took', time.time()-t1


        # t2 = time.time()
        # result = multiScatter_DS(num_el, material, E0, Emin, tilt, tables_moller, tables_gryz, Wc)
        # # each worker gets num_electrons/num_proc
        # print 'serial took', time.time()-t2
        # print result





#ray.get([rayScatter_cont.remote(material, num_el, E0, Emin, tilt) for i in range(num_el)])






    fileBSE = 'BSE_70tilt_ds.out'
    with open(fileBSE, 'w') as f:
        for proc in np.arange(num_proc-2):
            for item in BSEs[proc]:
                f.write("%s\n" % item)

    print 'BSE data was written to', fileBSE

    filepathlen = 'BS_pl_70tilt_ds.out'
    with open(filepathlen, 'w') as f:
        for proc in np.arange(num_proc-2):
            for item in pathlen[proc]:
                f.write("%s\n" % item)

    print 'BSE pathlen data was written to', filepathlen

    fileBS_BS = 'BS_theta_70tilt_ds.out'
    with open(fileBS_theta, 'w') as f:
        for proc in np.arange(num_proc-2):
            for item in thetas[proc]:
                f.write("%s\n" % item)

    print 'BS theta data was written to', fileBS_theta
#
#
# file_pos = 'positions_70tilt_ds.out'
# with open(file_pos, 'w') as f:
#         for item in position:
#             f.write("%s\n" % item)
#
# print 'pos data was written to', file_pos
#
#
#
# file_phiR = 'phiR_70tilt_ds.out'
# with open(file_phiR, 'w') as f:
#         for item in phi_R:
#             f.write("%s\n" % item)
#
# print 'phi angles for Rutherford was written to', file_phiR
#
#
# file_thetaR = 'thetaR_70tilt_ds.out'
# with open(file_thetaR, 'w') as f:
#         for item in theta_R:
#             f.write("%s\n" % item)
#
# print 'theta angles for Rutherford was written to', file_thetaR

#
# file_phiG = 'phiG_50tilt.out'
# with open(file_phiG, 'w') as f:
#         for item in phi_G:
#             f.write("%s\n" % item)
#
# print 'phi angles for Gryzinski was written to', file_phiG
#
# file_thetaG = 'thetaG_50tilt.out'
# with open(file_thetaG, 'w') as f:
#         for item in theta_G:
#             f.write("%s\n" % item)
#
# print 'theta angles for Gryzinski was written to', file_thetaG
#
# file_phiM = 'phiM_50tilt.out'
# with open(file_phiM, 'w') as f:
#         for item in phi_M:
#             f.write("%s\n" % item)
#
# print 'phi angles for Moller was written to', file_phiM
#
# file_thetaM = 'thetaM_50tilt.out'
# with open(file_thetaM, 'w') as f:
#         for item in theta_M:
#             f.write("%s\n" % item)
#
# print 'theta angles for Moller was written to', file_thetaM
