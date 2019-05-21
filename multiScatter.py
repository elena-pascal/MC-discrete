from multiprocessing import Pool, Process, Queue
import numpy as np
import os


from electron import electron
from singleScatter import singleScatter_DS

def saveResults(e, BSE_energy=[], BSE_pathlen=[], BSE_thetas=[]):
    '''
    For a given electron e, if e.outcome = BSE then save its
    escape energy, direction and scattering depth to lists
    '''

    if (e.outcome == 'backscattered'):
        BSE_energy.append(float(e.energy))
        BSE_pathlen.append(e_i.pathl[-10:])
        BSE_thetas.append(e_i.c2_halfTheta[-10:])
#        BSE_dir.append(e.dir)
    return (BSE_energy, BSE_pathlen, BSE_thetas)


def multiScatter_DS(num_el, material, E0, Emin, tilt, tables_moller, tables_gryz, Wc, parallel=False):
    # for parallel processes we need to make sure the random number seeds are different
    # for instance the process id
    if parallel:
        np.random.seed(os.getpid()) # only on Unix

    pos0 = np.array([0., 0., 0.,])
    dir0 = np.array([0., -np.sin(np.radians(tilt)) , np.cos(np.radians(tilt))])

    BSE_energy = []
    BSE_pathlen = []
    BSE_Thetas = []

    for _ in range(num_el):
        # start this electron
        e_i = electron(E0, pos0, dir0)
        # scatter until end of scatter
        singleScatter_DS(e_i, material, Emin, Wc, tables_moller, tables_gryz)

        saveResults(e_i, BSE_energy, BSE_pathlen, BSE_thetas)
        #BSE_energy.append(float(e_i.energy))
        #outcome.append(e_i.outcome)
    return (BSE_energy, BSE_pathlen, BSE_thetas)



# def model(n, multi=False, queue=0):
#     sum = 0
#     for x in range(1000):
#         sum += x*x
#     if multi:
#         queue.put(sum)
#     else:
#         return sum
#
# def multi_sim(CORES=4, T=100):
#     results = []
#     queues = [Queue() for i in range(CORES)]
#     args = [(int(T/CORES), True, queues[i]) for i in range(CORES)]
#     jobs = [Process(target=model, args=(a)) for a in args]
#     for j in jobs:
#         j.start()
#     for q in queues:
#         results.append(q.get())
#     for j in jobs:
#         j.join()
#
#     S = np.hstack(results)
#
#     return S
