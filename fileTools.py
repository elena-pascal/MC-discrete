#import h5py
import pandas as pd
import warnings
import numpy as np
####### read input file ##################
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

##########################################
def zipData(myData):
    '''
    Data arrives here as a list of tuple, a tuple for each thread.
    Zip the thread tuples together; returns lists of lists
    '''
    zippedParam = []

    for threadData in myData:
        for indx, param in enumerate(threadData[1]):
            if (indx < len(zippedParam)):
                zippedParam[indx].extend(list(param))
            else:
                zippedParam.append(list(param))

    #return tuple(tuple(param) for param in zippedParam)
    return zippedParam

def zipDict(dictA, dictB):
    '''
    For multiple dictionaries with same keys and of the form
    {'key' : [list]} merge the values in the same list
    '''
    for k in dictB.iterkeys():
        if k in dictA:
            dictA[k] += dictB[k]

    return dictA



######## write HDF5 file ##################
def writeBSEtoHDF5_old(data, input, filename, alpha, xy_PC, L):
    '''
    save all the backscattering relevant information to a structured data file

    filename = name of the file
    '''

    # if parallel the dataDictionary arrives here as a list of dictionaries
    # make one dict instead
    BSE_dict = data[0]['BSE']
    all_dict = data[0]['all']

    for dictionary in data[1:]:
        for k in dictionary.iterkeys():
            if (k == 'BSE'):
                zipDict(BSE_dict, dictionary[k])
            elif (k == 'all'):
                zipDict(all_dict, dictionary[k])
            else:
                print ('Not all data is saved to h5 file')


    # project directions on detector
    #onDet_pandasFrame = pd.DataFrame(projOnDetector(dataDictionary['direction'], alpha, xy_PC, L))

    # pandas doesn't like mixed elements entries, so I'm replacing the direction
    # arrays into three separate entries until I figure out something nicer
    BSEtable =  {'x_dir':[item[0] for item in BSE_dict['direction']],\
                 'y_dir':[item[1] for item in BSE_dict['direction']],\
                 'z_dir':[item[2] for item in BSE_dict['direction']]}

    del BSE_dict['direction']
    BSE_dict.update({'direction' : BSEtable})

    # write direction results to pandas data frame
    BSE_dir_df = pd.DataFrame(BSE_dict['direction'], columns=BSEtable.keys())

    # write energy results to pandas series
    BSE_e_s = pd.Series(BSE_dict['energy'])

    print ('---- number of BSEs:', len(BSE_dict['energy']))

    # write mean path length, total length travelled and number of
    # scattering events to pandas series
    all_mpl_s = pd.Series(all_dict['mean_pathl'])
    all_totalL_s = pd.Series(all_dict['total_path'])
    all_numScatt_s = pd.Series(all_dict['num_scatter'])

    # write on detector projection pandas to hdf5
    #onDet_pandasFrame.to_hdf(filename, key='onDet', mode='w')

    # write input parameters to pandas series
    input_s = pd.Series(input.values(), index=input.keys(), dtype=str)

    # TODO: don't supress all performance warnings though
    # pandas is going to complain about performace for the input string table
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    # HDFstore is a dict like object that reads and writes pandas with the PyTables library
    # pickled tables to be read with pandas:
    # pd.read_hdf(filename, 'BSE/directions')
    with pd.HDFStore(filename) as dataFile:
        # save some input parameters
        dataFile['input'] = input_s

        # write BSE energy and exit direction pandas data frame to hdf5
        dataFile['BSE/directions'] = BSE_dir_df
        dataFile['BSE/energy'] = BSE_e_s

        # write all electron information
        dataFile['all/mean_pathl'] = all_mpl_s
        dataFile['all/total_l'] = all_totalL_s
        dataFile['all/num_scatt'] = all_numScatt_s



def writeAllEtoHDF5(data, input, filename, alpha, xy_PC, L):
    '''
    save all the backscattering relevant information to a structured data file

    filename = name of the file
    '''

    data_list = zipData(data)

    # write direction results to pandas data frame
    allData_df = pd.DataFrame(np.array(data_list).T, columns=data[0][0] )

    print()
    # number of bascattered electrons is the length of any column for all 'backscattered' electrons
    print (' number of BSEs:', len(allData_df[allData_df.values == 'backscattered']['outcome']))

    # write input parameters to pandas series
    input_s = pd.Series(input.values(), index=input.keys(), dtype=str)

    # pandas is going to complain about performace for the input string table
    # warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    # HDFstore is a dict like object that reads and writes pandas with the PyTables library
    # pickled tables to be read with pandas:
    # pd.read_hdf(filename, 'BSE/directions')
    with pd.HDFStore(filename) as dataFile:
        # save some input parameters
        dataFile['input'] = input_s

        # write BSE energy and exit direction pandas data frame to hdf5
        dataFile['all_electrons'] = allData_df
