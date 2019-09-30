from extFunctions import gryz_dCS, moller_dCS
from tables import probTable



def genTables(inputPar, material):
    '''
    '''

    # define the Erange from input parameters
    Erange = (inputPar['Emin'], inputPar['E0'])

    # set chunk_size to whatever worked better on my machine
    csize = 100

    # generate Moller table
    mollerTable = probTable(type='Moller', shell='3s3p', func=moller_dCS,
                            E_range=Erange, Wmin=inputPar['Wc'],
                            tol_E=inputPar['tol_E'], tol_W=inputPar['tolW'],
                            material=material, mapTarget='tables', chunk_size=csize)
    mollerTable.generate()
    mollerTable.mapToMemory()


    # generate Gryzinski tables
    cumQuadInt_Gryz( inputPar['E0'], inputPar['Emin'],\
                                material.params['Es'], material.fermi_e,\
                                material.params['ns'], gryz_dCS,\
                                inputPar['num_BinsW'], inputPar['num_BinsE'] )

    # one table for each shell
    for Gshell in material.params['name_s']:
        gryzTable = probTable(type='Gryzinski', shell=Gshell, func=gryz_dCS,
                            E_range=Erange, Wmin=material['Es']['Gshell'],
                            tol_E=inputPar['tol_E'], tol_W=inputPar['tolW'],
                            material=material, mapTarget='tables', chunk_size=csize)
        gryzTable.generate()
        gryzTable.mapToMemory()




    # elif (inputPar['mode'] in ['diel', 'dielectric']):
    #     print ' ---- calculating dielectric function integral table'
    #     tables_diel =
