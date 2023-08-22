import torch
import argparse
from datetime import datetime
import os
import json
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Callable 
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import center_crop 

from cmrxrecon.dataloader.cmr_volume_dataset import CMR_volume_dataset
from cmrxrecon.dataloader.cmr_slice_dataset import CMR_slice_dataset
from cmrxrecon.dataloader.cmr_all_acceleration import All_Acceleration
from cmrxrecon.dataloader.cmr_all_modalities import All_Modalities
from cmrxrecon.transforms import normalize, unnormalize, phase_to_zero
from cmrxrecon.utils import convert_to_complex_batch, convert_to_complex, convert_to_real, rephase, crop_or_pad_to_size
from cmrxrecon.models.Unet import Unet
from cmrxrecon.losses import SSIMLoss


DATA_DIR = '/home/kadotab/projects/def-mchiew/kadotab/SpatialBasisNumpy/MultiCoil/Mapping/TrainingSet/'
OUTPUT_DIR = '/home/kadotab/projects/def-mchiew/kadotab/SpatialBasisCrop/'
def main():
    torch.manual_seed(0)
    
    weight_path = '/home/kadotab/scratch/runs/0821-13:33:15Unet_cmrxreconacc_10/model_weights/end.pt'
    acceleration_factor = '10'

    volume_dataset = CMR_volume_dataset(DATA_DIR, acc_factor=acceleration_factor, save_metadata=True)
    volume_dataset = All_Acceleration(volume_dataset)
    volume_dataset = All_Modalities(volume_dataset)
   
   
    # normalize and unormalize values
    norm = normalize(
        mean_input = torch.tensor([-3.7636e-06,  6.0398e-06,  5.1053e-06, -9.0448e-06,  5.2411e-06, 2.2193e-06]),
        std_input = torch.tensor([0.0026, 0.0026, 0.0026, 0.0026, 0.0026, 0.0026]),
    )
    unnorm = unnormalize(
                    torch.tensor([ 5.0149e-06, -4.1653e-06,  7.0542e-06, -7.9898e-06,  5.7021e-06, -4.0155e-06]),
                    torch.tensor([0.0026, 0.0026, 0.0026, 0.0026, 0.0026, 0.0026]),
                    )

    # U-net 
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = Unet(in_chan=6, out_chan=6, chans=64)
    net = net.to(device)

    # load model weights
    if device == 'cpu':
        net.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    else:
        net.load_state_dict(torch.load(weight_path))

    save_images(volume_dataset, net, True, norm, unnorm, device)

def save_images(volume_dataset, net, residual, normalize, unnorm, device):
    # loop through volumes
    for i in range(len(volume_dataset)):
        # get volumes
        input_volume, target_value, undersampled_fname = volume_dataset[i]

        # get modality and acceleration value
        modality = volume_dataset.dataset.dataset.modality
        acceleration = volume_dataset.dataset.dataset.R

        start = time.time()
        for slice_index in range(input_volume.shape[2]):
            input_slice = input_volume[:, :, slice_index, :]

            output = pass_model(net, normalize, unnorm, device, input_slice)
            input_volume[:, :, slice_index, :] = output

        print(time.time() - start)
        acceleration = acceleration.zfill(2)
        
        patient = undersampled_fname.split('/')[-2]
        dir = os.path.join(OUTPUT_DIR, 'AccFactor' + acceleration, patient)
        file_path = os.path.join(dir, 'spatial_basis_ml_' + modality + '.pt')

        if not os.path.exists(dir):
            try: 
                os.makedirs(dir)
            except OSError as e: 
                print(e)

        print(file_path)
        print(undersampled_fname)
        #torch.save(new_basis, file_path)


def pass_model(net, normalize, unnorm, device, input_slice):
    # move basis dim to 
    input_slice = input_slice.permute((2, 0, 1))
    #dummy target slice since not needed
    target_slice = torch.ones_like(input_slice) 

    # convert complex numbers to real 
    input_slice = convert_to_real(input_slice)
    target_slice = convert_to_real(target_slice)

    # normalize z-score and rephase to zero
    input_slice, _ = normalize((input_slice, target_slice)) 
    input_slice, _, input_phase = phase_to_zero(return_map=True)((input_slice, target_slice))

    input_slice = input_slice.to(device)

    # create batch dimension and convert to float
    input_slice = input_slice.unsqueeze(0).float()

    input_slice_crop = center_crop(input_slice, 128)

    output = net(input_slice_crop)
    output = input_slice_crop + output
    output = output.cpu()

    output = crop_or_pad_to_size(input_slice, output)

    output = unnorm(output)
    output = convert_to_complex_batch(output)
    output = rephase(output, input_phase)
    output = output.squeeze(0).permute((1, 2, 0))
    return output

if __name__ == '__main__':
    main()
