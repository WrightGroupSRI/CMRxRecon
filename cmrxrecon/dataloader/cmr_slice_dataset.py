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
            for volume in self.cmr_volume_loader.raw_data:
                slices.append(volume['slices'])

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
        
        (undersampled_basis, target_basis) = self.cmr_volume_loader[volume_index]

        undersampled_basis = undersampled_basis[:, :, slice_index, :]
        target_basis = target_basis[:, :, slice_index, :]

        # convert to real
        height, width, basis = undersampled_basis.shape
        undersampled_basis = torch.view_as_real(undersampled_basis).reshape((height, width, basis*2)).permute(2, 0, 1)
        target_basis = torch.view_as_real(target_basis).reshape((height, width, basis*2)).permute(2, 0, 1)

        # convert to float
        undersampled_basis = undersampled_basis.float()
        target_basis = target_basis.float()

        # normalize to 0-1
        # TODO: Test if normalizing to mean 0 std 1 is better. Test if gloabl or local noralization is better
        scaling_factor = undersampled_basis.abs().max()
        undersampled_basis /= scaling_factor
        target_basis /= scaling_factor

        #scaling_mean = undersampled_basis.mean((1, 2)).unsqueeze(-1).unsqueeze(-1)
        #scaling_std = undersampled_basis.std((1, 2)).unsqueeze(-1).unsqueeze(-1)
        #undersampled_basis = (undersampled_basis - scaling_mean)/scaling_std
        #target_basis = (target_basis - scaling_mean)/scaling_std

        assert not undersampled_basis.isnan().any()
        assert not target_basis.isnan().any()

        data = (undersampled_basis, target_basis, scaling_factor)
             
        if self.transforms:
            data = self.transforms(data)
        return data

