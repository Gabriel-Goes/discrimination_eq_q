# coding: utf-8

import glob
import os
import csv
import numpy as np
from numpy import moveaxis
import tensorflow as tf

import pandas as pd


def discrim(
        model: str,
        spectro_dir: str,
        output_dir: str,
        eventos: str,
        valid: bool) -> None:
    """
    Event class prediction.

    :param model_dir: Absolute path to the input trained model.
    :param spectro_dir: Absolute path to the input spectrograms.
    :param output_dir: Absolute path where to save to output files.
    """

    model = tf.keras.models.load_model(model)
    print(f'Number of events: {}')

    for i, (index, evento) in enumerate(eventos.groupby('Evento'), 1):
        print('*****************')
        print(f'EVENT {nb_evt} / {len(event_label)}')
        if valid:
            time = event_label[a][0]
            class_ = event_label[a][1]
        elif event_label[a].size == 2:
            time = event_label[a][0]
        else:
            time = event_label[a]
        pred_nat = 0
        pred_ant = 0

        list_spect = glob.glob(f'{spectro_dir}/{time}/*')
        print(f'Number of station: {len(list_spect)}')
        nb_st = 0

        for spect in list_spect:
            nb_st += 1
            print(f'Station {nb_st} / {len(list_spect)}', end="\r")
            file_name = (spect.split('/')[-1]).split('.npy')[0]
            station = file_name.split('_')[1]
            spect_file = np.load(f'{spect}', allow_pickle=True)
            spect_file = [np.array(spect_file)]

            x = moveaxis(spect_file, 1, 3)

            model_output = model.predict(x).round(3)
            pred = np.argmax(model_output, axis=1)

            if pred == 0:
                pred_final = 'Natural'
            if pred == 1:
                pred_final = 'Anthropogenic'

            if valid:
                predict_sta.writerow([
                    file_name,
                    station,
                    class_,
                    model_output[0][0],
                    model_output[0][1],
                    pred[0],
                    pred_final,
                ])
            else:
                predict_sta.writerow([
                    file_name,
                    station,
                    model_output[0][0],
                    model_output[0][1],
                    pred[0],
                    pred_final,
                ])

            pred_nat += model_output[0][0]
            pred_ant += model_output[0][1]

        pred_total = [pred_nat, pred_ant]
        try:
            pred_total = [
                (float(i) / sum(pred_total)).round(3) for i in pred_total
            ]
        except ZeroDivisionError:
            print(f'Erro Evento: {time}')
        pred_event = np.argmax(pred_total)
        if pred_event == 0:
            pred_final = 'Natural'
        if pred_event == 1:
            pred_final = 'Anthropogenic'

        if valid:
            predict_net.writerow([
                time,
                class_,
                pred_total[0],
                pred_total[1],
                pred_event,
                pred_final,
            ])
        else:
            predict_net.writerow([
                time,
                pred_total[0],
                pred_total[1],
                pred_event,
                pred_final,
            ])
