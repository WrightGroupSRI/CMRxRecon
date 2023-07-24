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
import cfl

from matio import loadmat, writemat

__all__ = ['espirit']

# Prepare data for BART processing
def prep_bart(fft_recon):
    dim = 6
    array = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims
        (np.expand_dims(np.expand_dims(np.transpose(np.expand_dims(fft_recon, axis=0), [0, 1, 2, 3, 5, 4]), axis=4), 
            axis=dim-1), axis=dim+1), axis=dim+1), axis=dim+1), axis=dim+1), axis=dim+1), axis=dim+1), axis=dim+1)
    return array

# Function to write reconstructed images and maps to .mat files
def write_recon(maps=None, espirit=None, path=None, contrast="T1"):
    out = {}
    out["maps"] = maps 
    out["recon"] = espirit
    writemat(key="maps", data=maps, path=os.path.join(path, f"{contrast}maps.mat"))
    writemat(key="recon", data=espirit, path=os.path.join(path, f"{contrast}recon_E.mat"))
    return True

# Function to estimate maps 
def maps(array):
    sh_ = np.shape(array)
    fft_b = prep_bart(array)
    m_ = np.zeros([1,sh_[0],sh_[1],sh_[2],sh_[3]], dtype=np.complex_)
    for i in range(0,sh_[-2]):
        # change to add gpu
        m_[...,i] = bart(1, 'ecalib -d3 -S -m1 -a -r1:48:9', fft_b[...,i])
    maps_ = prep_bart(np.transpose(m_, [1,2,3,4,0]))
    
  
    return maps_,fft_b


# Function for ESPIRiT reconstruction
def espirit(array, path=None, save_recon= None, iterations=50, contrast= 'T1'):
    maps_,fft_b = maps(array)
    # change to 100 iterations
    # change to add gpu
    # make sure regularizer is doing what we think it is
    recon_p = bart(1,f'pics -e -i {iterations} -d 5 -R W:6:0:0.05',fft_b,maps_)
    return np.transpose(np.squeeze(recon_p), axes=((0, 1, 3, 2)))
    
 

# Main function
if __name__ == '__main__':
    data_path = "/home/jay/condor/CMRxRecon/MultiCoil/Mapping/TrainingSet/AccFactor04/P001/T1map.mat"
    data = loadmat(key='kspace_sub04', path=data_path)
    espirit(data, path = f'{os.path.dirname(data_path)}/espirit_recon/', save_recon= True, contrast = os.path.basename(data_path)[0:2] )
