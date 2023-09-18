#!/usr/bin/env python3

###############################################################
# MAIN FUNCTION
# code that gets run in the docker container
#
# Calder Sheagren
# University of Toronto
# calder.sheagren@sri.utoronto.ca
# Date: June 26, 2023 
###############################################################

import os
import argparse
from datetime import datetime
import time
import numpy as np
import torch

from matio import *
from basis import *
from espirit import *
from cmrxrecon.models.Unet import Unet
from cmrxrecon.transforms import normalize, unnormalize, convert_to_real, phase_to_zero, center_crop
from cmrxrecon.utils import crop_or_pad_to_size, convert_to_complex_batch, rephase
from save_images import pass_model

def try_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

if __name__ == '__main__':

    start = datetime.now()

    print("Start:", start.strftime("%H:%M:%S"))
    parser = argparse.ArgumentParser(prog="DRUMs",
                                     description="Deep learning-Refined sUbspace Models\nCMRxRecon Sunnybrook Submission",
                                     epilog="Hope you enjoy using this!")

    parser.add_argument('--input_dir')
    parser.add_argument('--predict_dir')
    parser.add_argument('--weights_dir')

    args = parser.parse_args()

    assert(os.path.exists(args.input_dir)), "Please pass in a correct data location"

    pred_dir = try_dir(args.predict_dir)

    multicoil = try_dir(os.path.join(pred_dir, "MultiCoil"))
    mapping = try_dir(os.path.join(multicoil, "Mapping"))
    validation = try_dir(os.path.join(mapping, "ValidationSet"))

    data_dir = os.path.join(args.input_dir, "MultiCoil", "Mapping", "ValidationSet")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    net = Unet(in_chan=6, out_chan=6, chans=64).to(device)

    if device == 'cpu':
        net.load_state_dict(torch.load(args.weights_dir, map_location=torch.device('cpu')))
    else:
        net.load_state_dict(torch.load(args.weights_dir))

    # normalization performed before 
    norm = normalize(
        mean_input = torch.tensor([-3.7636e-06,  6.0398e-06,  5.1053e-06, -9.0448e-06,  5.2411e-06, 2.2193e-06]),
        std_input = torch.tensor([0.0026, 0.0026, 0.0026, 0.0026, 0.0026, 0.0026]),
    )
    unnorm = unnormalize(
        mean = torch.tensor([ 5.0149e-06, -4.1653e-06,  7.0542e-06, -7.9898e-06,  5.7021e-06, -4.0155e-06]).to(device),
        std = torch.tensor([0.0026, 0.0026, 0.0026, 0.0026, 0.0026, 0.0026]).to(device),
    )

    for a in ['04', '08', '10']:
        acc_factor = f'AccFactor{a}'

        acc_dir = os.path.join(data_dir, acc_factor)
        acc_dir_submission = try_dir(os.path.join(validation, acc_factor))

        pt_list = sorted(os.listdir(acc_dir))

        pt_num = 00

        while True:
            pt_num += 1

            max_num = int(pt_list[-1][1:])

            if pt_num > max_num:
                break

            pt_dir = f"P0{pt_num:02d}"

            pt_path = os.path.join(acc_dir, pt_dir)
            if not os.path.exists(pt_path):
                continue
            pt_path_submission = try_dir(os.path.join(acc_dir_submission, pt_dir))

            T1map = os.path.join(pt_path, 'T1map.mat') 
            T2map = os.path.join(pt_path, 'T2map.mat') 

            # Process T1 map, T2 map
            for path in [T1map, T2map]:
                if not os.path.exists(path):
                    continue
                data = loadmat(key=f'kspace_sub{a}', path=path)
                espirit_recon = espirit(data, iterations=100)
                SB, Svals, TB = spatial_temporal_basis(espirit_recon, L=3)

                SB_output_volume = np.zeros_like(SB)

                for slice_index in range(SB.shape[2]):
                    SB_slice = SB[:, :, slice_index, :]
                    SB_output = pass_model(net, norm, unnorm, device, torch.tensor(SB_slice).to(device)) 
                    SB_output_volume[:, :, slice_index, :] = SB_output.detach().cpu().numpy()

                SB = SB_output_volume
                imgs = outer_product(SB, Svals, TB)

                out = np.abs(imgs)
                # for testing set
                # out = np.transpose(np.abs(imgs), (3, 2, 1, 0))

                writemat(key=f'img4ranking', data=out, path=os.path.join(pt_path_submission, path.split('/')[-1]))
                # remove when we want to process T1 and T2 maps
                # break
            # end for [T1, T2]

            # remove when we want to process all patients
            # break
        # end for [pts]

        # remove when we want to process all acceleration factors
        # break
    # end for [acc factors]

    print("Start:", start.strftime("%H:%M:%S"))

    now = datetime.now()
    print("Now:", now.strftime("%H:%M:%S"))



