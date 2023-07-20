import os
from cmrxrecon.io import loadmat
import numpy as np

def main():
    """
    For some reason the .mat files are super slow to read using mat73 or hpy, so 
    I am converting them into numpy arrays and saving them. This reduces training time 
    by half.
    """

    directory = '/home/kadotab/projects/def-mchiew/kadotab/cmr_basis/'
    target_direcory = os.path.join(directory, 'kspace_single_full') 
    undersampled_directory4 = os.path.join(directory, 'kspace_single_sub' + '04') 
    undersampled_directory8 = os.path.join(directory, 'kspace_single_sub' + '08') 
    undersampled_directory10 = os.path.join(directory, 'kspace_single_sub' + '10') 
    target_files = os.listdir(target_direcory)

    print(f'Counting slices')
    for file in target_files:
        # create paths to files
        target_map_file = os.path.join(target_direcory, file, 'spatial_basis.mat')
        undersampled_map_file04 = os.path.join(undersampled_directory4, file, 'spatial_basis.mat')
        undersampled_map_file08 = os.path.join(undersampled_directory8, file, 'spatial_basis.mat')
        undersampled_map_file10 = os.path.join(undersampled_directory10, file, 'spatial_basis.mat')

        # load spatial basis
        spatial_basis_target = loadmat(path=target_map_file, key='spatial_basis')
        spatial_basis_04 = loadmat(path=undersampled_map_file04, key='spatial_basis')
        spatial_basis_08 = loadmat(path=undersampled_map_file08, key='spatial_basis')
        spatial_basis_10 = loadmat(path=undersampled_map_file10, key='spatial_basis')
        
        np.save(os.path.join(target_direcory, file, 'spatial_basis.npy'), spatial_basis_target)
        np.save(os.path.join(undersampled_directory4, file, 'spatial_basis.npy'), spatial_basis_04)
        np.save(os.path.join(undersampled_directory8, file, 'spatial_basis.npy'), spatial_basis_08)
        np.save(os.path.join(undersampled_directory10, file, 'spatial_basis.npy'), spatial_basis_10)
        
if __name__ == '__main__':
    main()
