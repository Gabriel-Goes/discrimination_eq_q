#! /usr/bin/env python3
# coding: utf-8

import argparse
import pandas as pd
from prediction import discrim
from data_process import spectro_extract

cs = 'ClassificadorSismologico/'
model = cs + 'fonte/rnc/model/model_2021354T1554.h5'
mseed = cs + 'arquivos/mseed'
spectro = cs + 'arquivos/espectro'
output = cs + 'arquivos/output'


def read_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument('--model',
                        type=str, default=model,
                        help="Model file path.")

    parser.add_argument('--mseed_dir',
                        type=str, default=mseed,
                        help="Input mseed file directory.")

    parser.add_argument('--spectro_dir',
                        type=str, default=spectro,
                        help='Output spectrogram file directory.')

    parser.add_argument('--output_dir',
                        type=str, default=output,
                        help='Output directory')

    parser.add_argument('--valid',
                        action="store_true",
                        help=' if the option "valid" is specified the \
                        validation mode will be applied. Csv input must have \
                        two columns (time, label_cat)')

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    eventos = pd.read_csv(
        'ClassificadorSismologico/arquivos/eventos/eventos.csv'
    )
    spectro_extract(
        mseed_dir=args.mseed_dir,
        spectro_dir=args.spectro_dir,
        eventos=eventos
    )
    discrim(
        model=args.model,
        spectro_dir=args.spectro_dir,
        output_dir=f'ClassificadorSismologico/files/output/{args.output_dir}',
        eventos=eventos,
        valid=args.valid
    )

    return


if __name__ == '__main__':
    args = read_args()
    main(args)
