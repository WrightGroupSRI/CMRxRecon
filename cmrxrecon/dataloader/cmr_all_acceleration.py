from torch.utils.data import Dataset, Subset
from cmrxrecon.dataloader.cmr_volume_dataset import CMR_volume_dataset

class All_Acceleration(Dataset):
    def __init__(self, volume_dataset: CMR_volume_dataset):
        super().__init__()

        if type(volume_dataset) is Subset:
            self.volume_dataset = volume_dataset.dataset
            self.dataset = volume_dataset
        else:
            self.volume_dataset = volume_dataset
            self.dataset = volume_dataset

    def __len__(self):
        return len(self.dataset) * 3

    def __getitem__(self, index):
        volume_index = index // 3
        acc_index = index % 3

        if acc_index == 0: 
            self.volume_dataset.set_acceleartion_factor('4')
        elif acc_index == 1:
            self.volume_dataset.set_acceleartion_factor('8')
        elif acc_index == 2:
            self.volume_dataset.set_acceleartion_factor('10')

        item = self.dataset[volume_index]
        return item
    
    def set_modality(self, modality):
        self.volume_dataset.set_modality(modality)


