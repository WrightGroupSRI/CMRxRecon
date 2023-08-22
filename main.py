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
import shutil
import argparse
import cfl
import matplotlib.pyplot as plt
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
    parser = argparse.ArgumentParser(prog="DSL-PILLAR",
                                     description="Deep Subspace Learning and Parallel Imaging with Temporal Low Rank Constraints\nCMRxRecon Sunnybrook Submission",
                                     epilog="Hope you enjoy using this!")

    parser.add_argument('--input_dir')
    parser.add_argument('--predict_dir')

    args = parser.parse_args()

    assert(os.path.exists(args.input_dir)), "Please pass in a correct data location"

    if os.path.exists(args.predict_dir):
        shutil.rmtree(args.predict_dir)
    pred_dir = try_dir(args.predict_dir)

    multicoil = try_dir(os.path.join(pred_dir, "MultiCoil"))
    mapping = try_dir(os.path.join(multicoil, "Mapping"))
    validation = try_dir(os.path.join(mapping, "TrainingSet"))

    data_dir = os.path.join(args.input_dir, "MultiCoil", "Mapping", "TrainingSet")

    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = Unet(in_chan=6, out_chan=6, chans=64)
    net.to(device)
    net.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    net.eval()

    # normalization performed before 
    norm = normalize(
        mean_input = torch.tensor([-3.7636e-06,  6.0398e-06,  5.1053e-06, -9.0448e-06,  5.2411e-06, 2.2193e-06]),
        std_input = torch.tensor([0.0026, 0.0026, 0.0026, 0.0026, 0.0026, 0.0026]),
    )
    unnorm = unnormalize(
        mean = torch.tensor([ 5.0149e-06, -4.1653e-06,  7.0542e-06, -7.9898e-06,  5.7021e-06, -4.0155e-06]).to(device),
        std = torch.tensor([0.0026, 0.0026, 0.0026, 0.0026, 0.0026, 0.0026].to(device)),
    )

    for a in ['04', '08', '10']:
        acc_factor = f'AccFactor{a}'

        acc_dir = os.path.join(data_dir, acc_factor)
        acc_dir_submission = try_dir(os.path.join(validation, acc_factor))

        pt_list = sorted(os.listdir(acc_dir))
        pt_num = 00

        while True:
            pt_num += 1

            pt_dir = f"P0{pt_num:02d}"

            pt_path = os.path.join(acc_dir, pt_dir)
            pt_path_submission = try_dir(os.path.join(acc_dir_submission, pt_dir))

            T1map = os.path.join(pt_path, 'T1map.mat') 
            T2map = os.path.join(pt_path, 'T2map.mat') 

            # Process T1 map, T2 map
            for path in [T1map]:#, T2map]:
                print(path)
                data = loadmat(key=f'kspace_sub{a}', path=path)
                espirit_recon = espirit(data, iterations=100)
                before = time.time()
                SB, Svals, TB = spatial_temporal_basis(espirit_recon, L=3)
                after = time.time()
                print("Basis fitting", after - before)

                before = time.time()
                print(f'Model inference')
                SB_output_volume = np.zeros_like(SB)
                for slice_index in range(SB.shape[2]):
                    SB_slice = SB[:, :, slice_index, :]
                    SB_output = pass_model(net, norm, unnorm, device, SB_slice) 
                    SB_output_volume[:, :, slice_index, :] = SB_output.squeeze(0)

                print(f'Model inference done: {time.time() - before}')

                SB = SB_output_volume
                imgs = outer_product(SB, Svals, TB)

                cfl.writecfl(f"espirit_recon_acc_{a}", espirit_recon)
                cfl.writecfl(f"outer_product_acc_{a}", imgs)
                cfl.writecfl(f"SB_acc_{a}", SB)
                cfl.writecfl(f"TB_acc_{a}", TB)


            # Export processed data
            os.system(f"touch {os.path.join(pt_path_submission, 'T1map.mat')}")
            os.system(f"touch {os.path.join(pt_path_submission, 'T2map.mat')}")

            # remove when we want to process all patients
            break

        # remove when we want to process all acceleration factors
        # break



