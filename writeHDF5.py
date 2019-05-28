import h5py
import pandas as pd



def writeBSEtoHDF5(data, filename):
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

    # pandas doesn't like mixed elements entries, so I'm replacing the direction
    # arrays into three separate entries until I figure out something nicer
    expandDir = {'x_dir':[item[0] for item in dataDictionary['direction']],\
                 'y_dir':[item[1] for item in dataDictionary['direction']],\
                 'z_dir':[item[2] for item in dataDictionary['direction']]}

    del dataDictionary['direction']
    dataDictionary.update(expandDir)
    # write results to pandas
    pandasFrame = pd.DataFrame(dataDictionary)

    # write pandas to hdf5
    pandasFrame.to_hdf(filename, key='BSE', mode='w')
