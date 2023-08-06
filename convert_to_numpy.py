import os
from cmrxrecon.io import loadmat
import numpy as np

def main():
    """
    For some reason the .mat files are super slow to read using mat73 or hpy, so 
    I am converting them into numpy arrays and saving them. This reduces training time 
    by half.
    """

    directory = '/home/kadotab/scratch/SpatialBasis/MultiCoil/Mapping/TrainingSet/'
    target_direcory = os.path.join(directory, 'FullSample') 
    undersampled_directory4 = os.path.join(directory, 'AccFactor' + '04') 
    undersampled_directory8 = os.path.join(directory, 'AccFactor' + '08') 
    undersampled_directory10 = os.path.join(directory, 'AccFactor' + '10') 
    target_files = os.listdir(target_direcory)

    print(f'Counting slices')
    for file in target_files:
        # create paths to files
        convert(undersampled_directory4, file, 'T1')
        convert(undersampled_directory8, file, 'T1')
        convert(undersampled_directory10, file, 'T1')
        convert(target_direcory, file, 'T1')

        convert(undersampled_directory4, file, 'T2')
        convert(undersampled_directory8, file, 'T2')
        convert(undersampled_directory10, file, 'T2')
        convert(target_direcory, file, 'T2')

def convert(directory, sample_folder, modality='T1'):
    file_name = 'spatial_basis_' + modality
    extension = '.mat'
    sample_directory = os.path.join(directory, sample_folder)
    directory = os.path.join(sample_directory, file_name + extension)

    new_extension = '.npy'
    basis = loadmat(path=directory, key='spatial_basis')

    np.save(os.path.join(sample_directory, file_name + new_extension), basis)
        
if __name__ == '__main__':
    main()
