
import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse
import os

import obspy as op
from obspy.clients.fdsn import Client as client
from obspy.core import Stream, read, UTCDateTime, Trace

import glob
import pandas as pd
import pickle

from data_process import spectro_extract_pred, spectro_extract_valid
from prediction import pred, valid 


def read_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode",
                      default="pred",
                      help="train/valid/pred")

    parser.add_argument("--model_dir",
                      default="./model/model_2021354T1554.h5",
                      help="Model file directory")

    parser.add_argument("--data_dir",
                      default="./mseed_demo",
                      help="Input mseed file directory")

    parser.add_argument("--csv_dir",
                      default=None,
                      help="If mode valid : Input csv file directory")

    parser.add_argument("--output_dir",
                      default='./output_demo',
                      help="Output directory")

    parser.add_argument("--save_spec",
                      default=True,
                      help="Save spectrograms")

    args = parser.parse_args()
    return args


def main(args):

    
    if args.mode == "pred":
        data = spectro_extract_pred(data_dir=args.data_dir)
        pred(model_dir=args.model_dir, spectro_dir='./spectro_demo', output_dir=args.output_dir)

    if args.mode == "valid":
        events = np.genfromtxt(f'{args.csv_dir}', delimiter=',', skip_header = 1, dtype=str)
        data = spectro_extract_valid(data_dir=args.data_dir, events_list=events)
        valid(model_dir=args.model_dir, spectro_dir='./spectro_demo', output_dir=args.output_dir, event_label=events)

    

    else:
        print("mode should be: train, valid, or pred")

    return


if __name__ == '__main__':
  args = read_args()
  main(args)
