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

from io import *
from basis import *
from espirit import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="DSL-PILLAR",
                                     description="Deep Subspace Learning and Parallel Imaging with Temporal Low Rank Constraints\nCMRxRecon Sunnybrook Submission",
                                     epilog="Hope you enjoy using this!")

    parser.add_argument('--input_dir')
    parser.add_argument('--predict_dir')

    args = parser.parse_args()

    print(args.input_dir, os.path.exists(args.input_dir), os.listdir(args.input_dir))
    print(args.predict_dir, os.path.exists(args.predict_dir), os.listdir(args.predict_dir))
