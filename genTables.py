from extFunctions import gryz_dCS, moller_dCS
from tables import probTable



def genTables(inputPar, material):
    '''
    '''

    # define the Erange from input parameters
    Erange = (inputPar['Emin'], inputPar['E0'])

    # set tolerance and chunk_size
    # note for these value you need at least 10G memory
    tolE = 5e-7
    tolW = 1e-7
    csize = 100

    # generate Moller table
    mollerTable = probTable(type='Moller', shell='3s3p', func=moller_dCS,
                            E_range=Erange, Wmin=inputPar['Wc'], tol_E=tolE, tol_W=tolW,
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
                            E_range=Erange, Wmin=material['Es']['Gshell'], tol_E=tolE, tol_W=tolW,
                            material=material, mapTarget='tables', chunk_size=csize)
        gryzTable.generate()
        gryzTable.mapToMemory()




    # elif (inputPar['mode'] in ['diel', 'dielectric']):
    #     print ' ---- calculating dielectric function integral table'
    #     tables_diel =
