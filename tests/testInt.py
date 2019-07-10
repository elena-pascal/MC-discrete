import numpy as np
import random
import bisect
import sys
import getopt

from material import material
from integrals import trapez_table
from extFunctions import gryz_dCS, moller_dCS

def main(argv):
    use_units = False
    inputfile = '../input.file'

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

def readInput(fileName):
    '''
    Read parameters from a given file.
    '''
    with open(fileName) as file:
        #split into lines
        rawdata = file.read().split('\n')

    data = {}

    for line in rawdata:
        if line.strip(): # ignore empty lines
            if not line.startswith("#"): # ignore comment lines
                line = line.partition('#')[0] # ignore in line comments
                parameter = line.split(":")

                # trim whitespaces
                param = [p.strip() for p in parameter]

                if (param[0] in ['mode', 'material', 'Bethe']):
                    # assign string to dictionary
                    data[param[0]] = param[1]
                elif ('Bins' in param[0]) or ('num_el' in param[0]):
                    # assign int to dictionary
                    data[param[0]] = int(param[1])
                else:
                    # assign float to dictionary
                    data[param[0]] = float(param[1])

    return data

# use units?
use_units, inputFile = main(sys.argv[1:])

# read input parameters
inputPar = readInput(inputFile)

# set material
thisMaterial = material(inputPar['material'])
print 'calculating moller table...'

tables_moller = trapez_table( inputPar['E0'], inputPar['Emin'],\
                              np.array([inputPar['Wc']]), thisMaterial.fermi_e,\
                              np.array([thisMaterial.params['n_val']]), moller_dCS,\
                              inputPar['num_BinsW'], inputPar['num_BinsE'] )

print 'table moller', tables_moller
Eidx_table = bisect.bisect_left(tables_moller[1], 28500)  # less then value
print 'Eidx', Eidx_table
int_enlosses_table = tables_moller[3][0, Eidx_table, :]
print 'int table',int_enlosses_table
print int_enlosses_table[-1]
integral = random.random() * int_enlosses_table[-1]
Wi_table = bisect.bisect_left(int_enlosses_table, integral)
E_loss = tables_moller[2][0, Eidx_table, :][Wi_table]

print E_loss
