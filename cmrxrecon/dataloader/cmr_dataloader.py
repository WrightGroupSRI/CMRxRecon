import os
import h5py
from typing import NamedTuple

import torch
import numpy as np
from torch.utils.data import Dataset
from cmrxrecon.dataloader import basis

class CMR_RawSample(NamedTuple):
    fname: os.PathLike
    slice_index: int    

class CMR_Dataloader(Dataset):
    def __init__(self, directory, transforms = None):
        super().__init__()
        directory = os.path.join(directory, '') # add trailing slash if not there
        self.data_files = os.listdir(directory)
        self.transforms = transforms

        self.raw_data = []
        for file in self.data_files:
            t1_map_file = os.path.join(directory, file, 'T1map.mat')
            with h5py.File(t1_map_file) as f:
                # file data dimensions reversed when reading using h5py
                file_data = f['kspace_single_full']
                ti, kz, ky, kx = file_data.shape
                for kz_slice in range(kz):
                    self.raw_data.append(CMR_RawSample(t1_map_file, kz_slice))
    

    def __getitem__(self, index):
        file_name, slice_index = self.raw_data[index]
        with h5py.File(file_name) as f:
            file_data = np.array(f['kspace_single_full'])
            slice_data = file_data[:, [slice_index], :, :]
            slice_data = np.transpose(slice_data, [3, 2, 1, 0])
            temporal_basis = basis.temporal_basis(slice_data)
            spatial_basis = basis.spatial_basis(slice_data, temporal_basis)
             
        if self.transforms:
            spatial_basis = self.transforms(spatial_basis)
        return spatial_basis

    def __len__(self):
        return len(self.raw_data)


if __name__ == '__main__':
    dataset = CMR_Dataloader('/home/kadotab/projects/def-mchiew/kadotab/SingleCoil/Mapping/TrainingSet/FullSample')
    dataset[0]
