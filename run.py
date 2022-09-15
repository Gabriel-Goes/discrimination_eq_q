#! /usr/bin/env python3
# coding: utf-8

import argparse

import numpy as np

from prediction import pred, valid
from data_process import spectro_extract


def read_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',
                        type=str, default='pred',
                        help='train/valid/pred')

    parser.add_argument('--model_dir',
                        type=str, default='./model/model_2021354T1554.h5',
                        help="Model file directory.")

    parser.add_argument('--data_dir',
                        type=str, default='./mseed_demo',
                        help="Input mseed file directory.")

    parser.add_argument('--spectro_dir',
                        type=str, default='./spectro_demo',
                        help='Output spectrogram file directory.')

    parser.add_argument('--csv_dir',
                        default=None,
                        help="If mode valid: Input csv file directory")

    parser.add_argument('--output_dir',
                        type=str, default='./output_demo',
                        help='Output directory')

    parser.add_argument("--save_spec",
                        default=True,
                        help="Save spectrograms")

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):

    if args.mode == 'pred':
        data = spectro_extract(data_dir=args.data_dir,
                               spectro_dir=args.spectro_dir)
        pred(model_dir=args.model_dir, spectro_dir=args.spectro_dir,
             output_dir=args.output_dir)

    elif args.mode == 'valid':
        events = np.genfromtxt(
            f'{args.csv_dir}', delimiter=',', skip_header=1, dtype=str)
        data = spectro_extract(data_dir=args.data_dir,
                               spectro_dir=args.spectro_dir, events_list=events)
        valid(model_dir=args.model_dir, spectro_dir='./spectro_demo',
              output_dir=args.output_dir, event_label=events)

    else:
        print("Mode should be: train, valid, or pred")

    return


if __name__ == '__main__':
    args = read_args()
    main(args)
