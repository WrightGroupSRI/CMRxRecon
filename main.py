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

from matio import *
from basis import *
from espirit import *

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
    validation = try_dir(os.path.join(mapping, "ValidationSet"))

    data_dir = os.path.join(args.input_dir, "MultiCoil", "Mapping", "ValidationSet")

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
            for path in [T1map, T2map]:
                data = loadmat(key=f'kspace_sub{a}', path=path)

                TB = temporal_basis(data) 
                espirit_recon = espirit(data)
                SB = spatial_basis(espirit_recon, TB)
                print(TB.shape, SB.shape)


            # Export processed data
            os.system(f"touch {os.path.join(pt_path_submission, 'T1map.mat')}")
            os.system(f"touch {os.path.join(pt_path_submission, 'T2map.mat')}")

            # remove when we want to process all patients
            break

        # remove when we want to process all acceleration factors
        break



