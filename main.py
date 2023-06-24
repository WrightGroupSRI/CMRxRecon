import numpy
import matplotlib
import os
import argparse

parser = argparse.ArgumentParser(prog="DL-LASER",
                                 description="CMRxRecon Sunnybrook Submission",
                                 epilog="Hope you enjoy using this!")

parser.add_argument('--input_dir')
parser.add_argument('--predict_dir')

args = parser.parse_args()
print(args.input_dir, os.path.exists(args.input_dir), os.listdir(args.input_dir))
print(args.predict_dir, os.path.exists(args.predict_dir), os.listdir(args.predict_dir))
