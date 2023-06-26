#!/usr/bin/env python3

###############################################################
# ESPIRiT RECONSTRUCTION
# functions to perform reconstruction
#
# Jaykumar Patel
# University of Toronto
# jaykumar.patel@mail.utoronto.ca
# Date: June 26, 2023 
###############################################################

from pyforest import *
from bart import bart
import scipy.io
import mat73
import cfl
import hdf5storage as hf

# Function to load MATLAB data
def loadmat(key=None, path=None):
    data = mat73.loadmat(path)
    array = data.get(key)
    return array
    
# Prepare data for BART processing
def prep_bart(fft_recon):
    dim = 6
    array = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims
        (np.expand_dims(np.expand_dims(np.transpose(np.expand_dims(fft_recon, axis=0), [0, 1, 2, 3, 5, 4]), axis=4), 
            axis=dim-1), axis=dim+1), axis=dim+1), axis=dim+1), axis=dim+1), axis=dim+1), axis=dim+1), axis=dim+1)
    return array

# Function to write data to MATLAB .mat file
def writemat(key=None, data=None, path=None):
    assert(key), "Please pass in key"
    assert(data.ndim > 0), "Please pass in data"
    assert(path), "Please pass in path"
    hf.savemat(path, {key:data}, appendmat=False)
    return path

# Function to write reconstructed images and maps to .mat files
def write_recon(maps=None, espirit=None, path=None, contrast="T1"):
    out = {}
    out["maps"] = maps 
    out["recon"] = espirit
    writemat(key="maps", data=maps, path=os.path.join(path, "maps.mat"))
    writemat(key="recon", data=espirit, path=os.path.join(path, "recon_E.mat"))
    return True

# Function to estimate maps 
def maps(array):
    sh_ = np.shape(array)
    fft_b = prep_bart(array)
    m_ = np.zeros([1,sh_[0],sh_[1],sh_[2],sh_[3]])
    for i in range(0,sh_[-2]):
        m_[...,i] = bart(1, 'ecalib -d3 -g -S -m1 -a -r1:48:9', fft_b[...,i])
    maps_ = prep_bart(np.transpose(m_, [1,2,3,4,0]))
    
  
    return maps_,fft_b


# Function for ESPIRiT reconstruction
def recon_espirit(array, path=None, save_recon= None):
    maps_,fft_b = maps(array)
    recon_p = bart(1,'pics -e  -g -i 100 -R W:6:0:0.05',fft_b,maps_)
    
    if save_recon==True:
        if not os.path.exists(f"{path}/espirit_recon/"):
            os.makedirs(f"{path}/espirit_recon/")
        write_recon(maps=maps_, espirit= np.flip(recon_p, axis = [1,2]), path=path)


# Main function
if __name__ == '__main__':
    data_path = "/hdd/Data/CMRxRecon/SingleCoil/Mapping/TrainingSet/FullSample/P001/T1map.mat"
    data = loadmat(key='kspace_full', path=data_path)
    recon_espirit(data, path = os.path.dirname(data_path), save_recon= True)