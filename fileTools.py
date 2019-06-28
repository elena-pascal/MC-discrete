import h5py
import pandas as pd

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


######## write HDF5 file ##################
def writeBSEtoHDF5(data, filename, alpha, xy_PC, L):
    '''
    save all the backscattering relevant information to a structured data file

    filename = name of the file
    '''

    def zipDict(dictA, dictB):
        '''
        For multiple dictionaries with same keys and of the form
        {'key' : [list]} merge the values in the same list
        '''
        for k in dictB.iterkeys():
            if k in dictA:
                dictA[k] += dictB[k]
        return dictA

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
                print 'Not all data is saved to h5 file'


    # project directions on detector
    #onDet_pandasFrame = pd.DataFrame(projOnDetector(dataDictionary['direction'], alpha, xy_PC, L))

    # pandas doesn't like mixed elements entries, so I'm replacing the direction
    # arrays into three separate entries until I figure out something nicer
    expandDir = {'x_dir':[item[0] for item in BSE_dict['direction']],\
                 'y_dir':[item[1] for item in BSE_dict['direction']],\
                 'z_dir':[item[2] for item in BSE_dict['direction']]}

    del BSE_dict['direction']
    BSE_dict.update(expandDir)


    print '---- number of BSEs:', len(BSE_dict['energy'])

    # write results to pandas
    BSE_pandasFrame = pd.DataFrame(BSE_dict).T

    # write BSE energy and exit direction pandas to hdf5
    BSE_pandasFrame.to_hdf(filename, key='BSE', mode='w')


    # write results to pandas
    all_pandasFrame = pd.DataFrame(all_dict).T

    # write BSE energy and exit direction pandas to hdf5
    all_pandasFrame.to_hdf(filename, key='all_e', mode='w')

    # write on detector projection pandas to hdf5
    #onDet_pandasFrame.to_hdf(filename, key='onDet', mode='w')
