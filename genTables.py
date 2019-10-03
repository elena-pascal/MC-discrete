from extFunctions import gryz_dCS, moller_dCS
from tables import probTable



def genTables(inputPar, material):
    '''
    '''

    # define the Erange from input parameters
    Erange = (inputPar['Emin'], inputPar['E0'])

    # set chunk_size to whatever worked better on my machine
    csize = 100

    # instance for Moller table
    mollerTable = probTable(type='Moller', shell=material.params['name_val'], func=moller_dCS,
                            E_range=Erange,
                            tol_E=inputPar['tol_E'], tol_W=inputPar['tol_W'],
                            material=material, mapTarget='tables', chunk_size=csize,
                            Wc=inputPar['Wc'])

    # generate Moller table
    mollerTable.generate()

    # map to memory
    mollerTable.mapToMemory()



    # one Gryzinki table for each shell
    for Gshell in material.params['name_s']:
        # instance for Gryzinski table
        gryzTable = probTable(type='Gryzinski', shell=Gshell, func=gryz_dCS,
                            E_range=Erange,
                            tol_E=inputPar['tol_E'], tol_W=inputPar['tol_W'],
                            material=material, mapTarget='tables', chunk_size=csize)

        # generate Gryzinski table for shell Gshell
        gryzTable.generate()

        # map to memory
        gryzTable.mapToMemory()




    # elif (inputPar['mode'] in ['diel', 'dielectric']):
    #     print ' ---- calculating dielectric function integral table'
    #     tables_diel =
