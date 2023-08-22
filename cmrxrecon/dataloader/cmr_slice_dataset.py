import os
from typing import NamedTuple
import numpy as np

import torch
from torch.utils.data import Dataset
from cmrxrecon.dataloader.cmr_volume_dataset import CMR_volume_dataset


class CMR_slice_dataset(Dataset):
    def __init__(self, cmr_volume_loader:CMR_volume_dataset, transforms: callable = None):
        super().__init__()

        self.cmr_volume_loader = cmr_volume_loader
        self.transforms = transforms

        self.slice_data = []
        print(f'Counting Slices')
        slices = []

        if type(self.cmr_volume_loader) is torch.utils.data.Subset:
            self.sampled_indecies = self.cmr_volume_loader.indices
            for index in self.sampled_indecies:
                volume = self.cmr_volume_loader.dataset.raw_data[index]
                slices.append(volume['slices'])
            self.slice_cumlative_sum = np.cumsum(slices)
        else:
            for volume in range(len(self.cmr_volume_loader)):
                slices.append(self.cmr_volume_loader[volume][0].shape[2])

            self.slice_cumlative_sum = np.cumsum(slices)
    
        print(f'Counted {self.slice_cumlative_sum[-1]} slices')
    
    def __len__(self):
        return self.slice_cumlative_sum[-1]

    def __getitem__(self, index) -> torch.Tensor:
        # convert current index to volume index
        volume_index = np.where(self.slice_cumlative_sum > index)[0][0]
        # get the slice index
        if volume_index > 0:
            slice_index = index - self.slice_cumlative_sum[volume_index - 1] 
        else: 
            slice_index = index

        # load volume and get slice 
        undersampled_basis, target_basis, _ = self.cmr_volume_loader[volume_index]
        undersampled_basis = undersampled_basis[:, :, slice_index, :]
        target_basis = target_basis[:, :, slice_index, :]

        # convert to real shape is now basis, height, width
        height, width, basis = undersampled_basis.shape
        #undersampled_basis = torch.view_as_real(undersampled_basis).reshape((height, width, basis*2))
        #target_basis = torch.view_as_real(target_basis).reshape((height, width, basis*2))

        undersampled_basis = undersampled_basis.permute(2, 0, 1)
        target_basis = target_basis.permute(2, 0, 1)

        
        assert not undersampled_basis.isnan().any()
        assert not target_basis.isnan().any()

        data = (undersampled_basis, target_basis)
             
        if self.transforms:
            data = self.transforms(data)
        return data

