#!/usr/bin/env python3

###############################################################
# IO FUNCTIONS 
# for dealing with mat73 files
# 
#
# Calder Sheagren
# University of Toronto
# calder.sheagren@sri.utoronto.ca
# Date: May 29, 2023 
###############################################################

#import mat73
import hdf5storage as hf
import h5py
import numpy as np

__all__ = ['loadmat', 'writemat']

def loadmat(key=None, path=None):
    assert(path), "Please pass in filepath"
    with h5py.File(path) as f:
        if key:
            data = f[key]
        else:
            print(f.keys())
        ndim = data.ndim
        data = np.transpose(data, np.arange(ndim-1, -1, -1))
        data = data['real'] + 1j * data['imag']

    # data = mat73.loadmat(path)
    # array = data.get(key)
    return data

def writemat(key=None, data=None, path=None):
    assert(key), "Please pass in key"
    assert(data.ndim > 0), "Please pass in data"
    assert(path), "Please pass in path"
    hf.savemat(path, {key:data}, appendmat=False)
    return path
  
if __name__ == '__main__':
    fully_sampled_path = "/hdd/Data/CMRxRecon/SingleCoil/Mapping/TrainingSet/FullSample/P001/T1map.mat"
    data = loadmat(key='kspace_single_full', path=fully_sampled_path)

    writepath = writemat(key='kspace_single_full', data=data, path = "/hdd/Data/CMRxRecon/SingleCoil/Mapping/TrainingSet/FullSample/P001/T1map_write.mat")
    data_read = loadmat(key='kspace_single_full', path=writepath)
    print(np.allclose(data, data_read))

