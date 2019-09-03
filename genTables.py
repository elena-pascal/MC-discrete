from extFunctions import gryz_dCS, moller_dCS
from integrals import cumQuadInt_Gryz, cumQuadInt_Moller
import time

def genTables(inputPar, material):
    '''
    '''
    print ('---- calculating Moller tables')
    start = time.time()
    cumQuadInt_Moller( inputPar['E0'], inputPar['Emin'],\
                                   inputPar['Wc'], material.fermi_e,\
                                   material.params['n_val'], moller_dCS,\
                                   inputPar['num_BinsW'], inputPar['num_BinsE'] )

    print ('moller tables took', time.time()-start)

    print ('---- calculating Gryzinski tables')
    start = time.time()
    cumQuadInt_Gryz( inputPar['E0'], inputPar['Emin'],\
                                material.params['Es'], material.fermi_e,\
                                material.params['ns'], gryz_dCS,\
                                inputPar['num_BinsW'], inputPar['num_BinsE'] )

    print ('gryz tables took', time.time()-start)
    
    # elif (inputPar['mode'] in ['diel', 'dielectric']):
    #     print ' ---- calculating dielectric function integral table'
    #     tables_diel =
