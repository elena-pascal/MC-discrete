import h5py
import pandas as pd

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
                
                if ((param[0] == 'mode') or (param[0] == 'material')):
                    # assign string to dictionary
                    data[param[0]] = param[1]
                elif ('Bins' in param[0]):
                    # assign int to dictionary
                    data[param[0]] = int(param[1])
                else:
                    # assign float to dictionary
                    data[param[0]] = float(param[1])

    return data


def writeBSEtoHDF5(data, filename, alpha, xy_PC, L):
    '''
    save all the backscattering relevant information to a structured data file

    filename = name of the file
    '''
    # if parallel the dataDictionary arrives here as a list of dictionaries
    dataDictionary = {}
    for k in data[0].iterkeys():
        dataDictionary[str(k)] = []
        for d in data:
            dataDictionary[str(k)] += d[str(k)]

    # project directions on detector
    onDet_pandasFrame = pd.DataFrame(projOnDetector(dataDictionary['direction'], alpha, xy_PC, L))

    # pandas doesn't like mixed elements entries, so I'm replacing the direction
    # arrays into three separate entries until I figure out something nicer
    expandDir = {'x_dir':[item[0] for item in dataDictionary['direction']],\
                 'y_dir':[item[1] for item in dataDictionary['direction']],\
                 'z_dir':[item[2] for item in dataDictionary['direction']]}

    del dataDictionary['direction']
    dataDictionary.update(expandDir)

    print '---- number of BSEs:', len(dataDictionary['energy'])

    # write results to pandas
    BSE_pandasFrame = pd.DataFrame(dataDictionary).T

    # write BSE energy and exit direction pandas to hdf5
    BSE_pandasFrame.to_hdf(filename, key='BSE', mode='w')

    # write on detector projection pandas to hdf5
    onDet_pandasFrame.to_hdf(filename, key='onDet', mode='w')
