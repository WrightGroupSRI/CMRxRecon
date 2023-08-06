from torch.utils.data import Dataset, Subset


class All_Modalities(Dataset):
    def __init__(self, volume_dataset):
        super().__init__()

        if type(volume_dataset) is Subset:
            self.volume_dataset = volume_dataset.dataset
            self.dataset = volume_dataset
        else:
            self.volume_dataset = volume_dataset
            self.dataset = volume_dataset

    def __len__(self):
        return len(self.dataset) * 2

    def __getitem__(self, index):
        volume_index = index // 2
        modality_index = index % 2

        if modality_index == 0:
            self.volume_dataset.set_modality('T1')
        elif modality_index == 1:
            self.volume_dataset.set_modality('T2')

        return self.dataset[volume_index]

    def set_acceleartion_factor(self, new_factor):
        self.volume_dataset.set_acceleration_factor(new_factor)
        