import os
import json 
import numpy as np

import torch
from torch.utils.data import Dataset


from cmrxrecon.io import loadmat

class CMR_volume_dataset(Dataset):
    """Volume dataset for MICCAI challenge. Takes in directory and loads volumes of 
    spatial basis. For some reason, much faster in numpy saved format so using that now
    
    """

    def __init__(self, directory:os.PathLike, acc_factor:str = 4, transforms: callable = None, save_metadata = False, meta_data: os.PathLike = ""):
        super().__init__()
        assert acc_factor in [4, 8, 10], 'Acceleration factor should be 4, 8, or 10'
        self.R = acc_factor

        directory = os.path.join(directory, "") # add trailing slash if not there
        target_direcory = os.path.join(directory, 'kspace_single_full') 
        undersampled_directory04 = os.path.join(directory, 'kspace_single_sub04')
        undersampled_directory08 = os.path.join(directory, 'kspace_single_sub08')
        undersampled_directory10 = os.path.join(directory, 'kspace_single_sub10')

        self.target_files = os.listdir(target_direcory)
        self.undersampled_files04 = os.listdir(undersampled_directory04)
        self.undersampled_files08 = os.listdir(undersampled_directory08)
        self.undersampled_files10 = os.listdir(undersampled_directory10)
        self.transforms = transforms

        self.raw_data = []
        if meta_data != "":
            with open(meta_data) as f:
                self.raw_data = json.load(f)
        else:
            print(f'Counting Volumes')
            for file in self.target_files:

                # create paths to files
                target_map_file = os.path.join(target_direcory, file, 'spatial_basis.npy')
                undersampled_map_file04 = os.path.join(undersampled_directory04, file, 'spatial_basis.npy')
                undersampled_map_file08 = os.path.join(undersampled_directory08, file, 'spatial_basis.npy')
                undersampled_map_file10 = os.path.join(undersampled_directory10, file, 'spatial_basis.npy')

    
                # load spatial basis
                if target_map_file.endswith('.mat'):
                    spatial_basis_target = loadmat(path=target_map_file, key='spatial_basis')
                elif target_map_file.endswith('.npy'):
                    spatial_basis_target = np.load(target_map_file)
                else:
                    raise ValueError(f'Cannot read extension of type {target_map_file.split(".")[-1]}')
                _, _, z_slices, _ = spatial_basis_target.shape

                # create dictionary of files for a specific sample
                slice_objects = {
                    'target_fname': target_map_file, 
                    'undersampled_fname_04': undersampled_map_file04, 
                    'undersampled_fname_08': undersampled_map_file08,
                    'undersampled_fname_10': undersampled_map_file10,
                    'slices': z_slices
                    }
                self.raw_data.append(slice_objects)

        if save_metadata:
            with open('header.json', 'w') as f:
                f.write(json.dumps(self.raw_data, indent=4))


        print(f'Counted {len(self.raw_data)} volumes')
    
    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index) -> torch.Tensor:
        file_dict = self.raw_data[index]
        target_fname = file_dict['target_fname']

        if self.R == 4:
            undersampled_fname = file_dict['undersampled_fname_04']
        elif self.R == 8:
            undersampled_fname = file_dict['undersampled_fname_08']
        elif self.R == 10:
            undersampled_fname = file_dict['undersampled_fname_10']
        else: 
            raise ValueError(f'The acceleartion factor should be 4, 8 or 10 but got {self.R}')

        # load data
        if target_fname.endswith('.npy'):
            target_basis = np.load(target_fname)
            undersampled_basis = np.load(undersampled_fname)
        elif target_fname.endswith('.mat'):
            target_basis = loadmat(path=target_fname, key='spatial_basis')
            undersampled_basis = loadmat(path=undersampled_fname, key='spatial_basis')

        target_basis = torch.from_numpy(target_basis[:, :, :, :3])
        undersampled_basis = torch.from_numpy(undersampled_basis[:, :, :, :3])

        assert not undersampled_basis.isnan().any()
        assert not target_basis.isnan().any()

        return (undersampled_basis, target_basis)

if __name__ == '__main__':
    CMR_volume_dataset('/home/kadotab/projects/def-mchiew/kadotab/cmr_basis/')