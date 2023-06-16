import os

import torch
from torch.utils.data import Dataset

class CMR_Dataloader(Dataset):
    def __init__(self, directory, transforms = None):
        super().__init__()
        self.data_files = os.listdir(directory)
        self.transforms = transforms
    
    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.data_files)