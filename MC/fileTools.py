#import h5py
import pandas as pd
import warnings
import numpy as np
####### read input file ##################
def readInput(fileName='input.file'):
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

                if (param[0] in ['mode', 'material', 'Bethe_model']):
                    # assign string to dictionary
                    data[param[0]] = param[1]

                elif (param[0] in ['num_el', 'maxScatt']):
                    # assign int to dictionary
                    data[param[0]] = int(param[1])

                elif ('output' in param[0]):
                    # make dictionary with output values
                    data[param[0]] = param[1].replace(" ", "").split(",")

                elif ('gen_tables' in param[0]):
                    # set boolean parameter
                    data[param[0]] = True if 'es' in param[1] else False
                    
                else:
                    # assign float to dictionary
                    data[param[0]] = float(param[1])

    return data

##########################################
def zipData(myData):
    '''
    Data arrives here as a list of tuple, a tuple for each thread,
    for each entry in data dictionary
    Zip the thread tuples together; returns list of lists instead
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
    for k in dictB.keys():
        if k in dictA:
            dictA[k] += dictB[k]
        else:
            dictA[k] =  dictB[k]

    return dictA



######## write HDF5 file ##################
def writeBSEtoHDF5(results, input, filename):
    '''
    Save the results to structured data file.
    If parallel, the result dictionary arrives here from the multiprocessing Queue,
    therefore we need to zip the list of lists of dictionaries into single dictionaries

    ie. I made an overcomplicated results format and now I have to sort it

    input:
        results  : {'electrons' : [list], 'scatterings' : [list]}
        input    : input data
        filename : name of the file
    '''

    dataset = {}

    # for every dataset in results
    for dataset_key in results.keys():
        dataset[dataset_key] = {}

        # for every dictionary in the list of dictionaries
        for dictionary in results[dataset_key]:
            # zip the dictionaries together
            dataset[dataset_key] = zipDict(dataset[dataset_key], dictionary)


    # project directions on detector
    #onDet_pandasFrame = pd.DataFrame(projOnDetector(dataDictionary['direction'], alpha, xy_PC, L))

    # pandas doesn't like mixed elements entries, so I'm replacing the direction
    # arrays into three separate entries until I figure out something nicer
    # BSEtable =  {'x_dir':[item[0] for item in BSE_dict['direction']],\
    #              'y_dir':[item[1] for item in BSE_dict['direction']],\
    #              'z_dir':[item[2] for item in BSE_dict['direction']]}
    #
    # del BSE_dict['direction']
    # BSE_dict.update({'direction' : BSEtable})

    # write direction results to pandas data frame
    #BSE_dir_df = pd.DataFrame(BSE_dict['direction'], columns=BSEtable.keys())

    # make pandas dataframes from dataset dictionaries and save to HDF5
    with pd.HDFStore(filename) as dataFile:
        for dataset_key in dataset.keys():
            # make pandas dataframe
            df = pd.DataFrame.from_dict(dataset[dataset_key], dtype=float)

            # save to HDF5
            dataFile[dataset_key] = df

        # write input parameters to pandas series -> transposed dataframe
        dataFile['input'] = pd.DataFrame(pd.Series(input)).T

    # pandas is going to complain about performace for the input string table
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


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



#### output dictionary class
class thingsToSave:
    """Parameters to save from the simulation"""

    def __init__(self, args):
        ''' An unknown number of parameters are passed in here.
            Set up a dictionary which will contain the lists of values
            for every argument we want to track'''
        self.dict = {}
        for arg in args:
            self.dict[arg] = []

    def addToList(self, par, value):
        '''
        Add value to the parameter list only if the parameter
        is in the list of parameters we want to save
        '''
        if (par in self.dict.keys()):
            self.dict[par].append(value)
