import unittest

from multiprocessing import Process, cpu_count, Queue

from multiscatter import scatterMultiEl_cont, scatterMultiEl_DS
from material import material

# test the Monte Carlo results match the underlying probability ditribution from which events are sampled

class TestScatterAnglesforDS(unitttest.TestCase):
    def SetUp(self):
        num_proc = cpu_count()-1
        output = Queue()

        processes = [Process(target=scatterMultiEl_DS, args=(num_el=5000, material=material('Al'),
                                                            E0=20000, Emin=500,
                                                            tilt=0, table_moller=table_moller,
                                                            tables_gryz=tables_gryz, Wc=50,
                                                            output=output, count=count)) for count in range(num_proc)]

        # start threads
        for p in processes:
            p.start()

        result =  [pickle.loads(output.get()) for p in processes]

        # save to file
        fileBSE = 'testData/testAngles_DS.temp'

        from parameters import alpha, xy_PC, L
        projection = (alpha, xy_PC, L)
        testValues = ('mollerAngle')
        writeAllEtoHDF5(result, inputPar, fileBSE, *projection, **testVals)
