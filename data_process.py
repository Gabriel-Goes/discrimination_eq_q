# #!/home/ipt/.pyenv/versions/sismologia/bin/python
# coding: utf-8
# Author: Gabriel Góes Rocha de Lima
# CoAuthor: Lucas Schirbel
# Date: 2021-07-01
# Version: 0.1.0

# Description: This script contains the functions used to process the data

# ---------------------------- IMPORT LIBRARIES ----------------------------- #
import os
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd

import obspy as op
from obspy.signal.invsim import cosine_taper
from obspy.signal.trigger import classic_sta_lta
# import matplotlib.pyplot as plt
# from obspy.core import read
# from obspy.signal.trigger import plot_trigger


# ---------------------------- FUNCTION DEFINITIONS ------------------------- #
def stride_windows(x, n, noverlap=None, axis=0):
    if noverlap is None:
        noverlap = 0

    if noverlap >= n:
        raise ValueError('noverlap must be less than n')
    if n < 1:
        raise ValueError('n cannot be less than 1')

    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError('only 1-dimensional arrays can be used')
    if n == 1 and noverlap == 0:
        if axis == 0:
            return x[np.newaxis]
        else:
            return x[np.newaxis].transpose()
    if n > x.size:
        raise ValueError('n cannot be greater than the length of x')

    # np.lib.stride_tricks.as_strided easily leads to memory corruption for
    # non integer shape and strides, i.e. noverlap or n. See #3845.
    noverlap = int(noverlap)
    n = int(n)

    step = n - noverlap
    if axis == 0:
        shape = (n, (x.shape[-1]-noverlap)//step)
        strides = (x.strides[0], step*x.strides[0])
    else:
        shape = ((x.shape[-1]-noverlap)//step, n)
        strides = (step*x.strides[0], x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def fft_taper(data: np.ndarray) -> np.ndarray:
    return data * cosine_taper(npts=data.size, p=0.2)


def get_fft(
        trace: op.core.trace.Trace,
        WINDOW_LENGTH: int,
        OVERLAP: float,
        nb_pts: int) -> tuple:
    s_rate = trace.stats.sampling_rate
    nb_pts = int(WINDOW_LENGTH * s_rate)
    nb_overlap = int(OVERLAP * nb_pts)
    window = fft_taper(np.ones(nb_pts, trace.data.dtype))
    result = stride_windows(trace.data, nb_pts, nb_overlap, axis=0)
    result = mlab.detrend(result, mlab.detrend_linear, axis=0)
    result = result * window.reshape((-1, 1))
    numFreqs = nb_pts // 2 + 1
    result = np.fft.fft(result, n=nb_pts, axis=0)[:numFreqs, :]
    freqs = np.fft.fftfreq(nb_pts, 1 / s_rate)[:numFreqs]
    freqs[-1] *= -1
    result = result[1:]
    freqs = freqs[1:]
    result = np.abs(result) / trace.data.size
    result = result.ravel()
    return result, freqs


def calculate_cft(stream):
    st = stream.copy()
    cft_array = np.array([])
    for tr in st:
        cft = classic_sta_lta(
            tr.data,
            int(2 * tr.stats.sampling_rate),
            int(10 * tr.stats.sampling_rate)
        )
        cft_array = np.append(cft_array, cft)
    cft_array = cft_array.reshape(3, 6001)

    return cft_array


def cft_max(
            cft_array: np.ndarray,
            n: float = 1.7
        ):
    for cft in cft_array:
        if cft.max() > n:
            return cft.max()

    return None


def spectro_extract(
        mseed_dir: str,
        spectro_dir: str,
        eventos: pd.DataFrame) -> None:
    WINDOW_LENGTH = 1
    OVERLAP = (1 - 0.75)
    eventos['Compo'] = [[] for _ in range(len(eventos))]
    eventos['Error'] = [[] for _ in range(len(eventos))]
    eventos['Warning'] = [[] for _ in range(len(eventos))]
    eventos.reset_index(inplace=True)
    eventos.set_index(['Event', 'Station'], inplace=True)
    eventos.sort_index(inplace=True)

    n_ev = eventos.groupby(level=0).size().shape[0]
    print(f'Number of events: {n_ev}')

    for i, (ev_index, evento) in enumerate(eventos.groupby(level=0), start=1):
        print('*****************')
        print(f'EVENT: {ev_index} ({i} / {n_ev})')
        print(f' - Number of picks: {evento.shape[0]}')

        for j, (pk_index, pick) in enumerate(evento.groupby(level=1), start=1):
            print(f'PICK: {pk_index} ({j} / {evento.shape[0]})')
            if pick.shape[0] != 1:
                err = f' - Error! pick.shape[0] != 1 ({pick.shape[0]})'
                eventos.loc[(ev_index, pk_index), 'Error'] = err
                print(err)
                continue
            p_path = pick.Path.values[0]
            st = op.read(f'arquivos/mseed/{p_path}', dtype=float)
            st.detrend('demean')
            st.taper(0.05)
            st = st.filter('highpass', freq=2, corners=4, zerophase=True)
            if st[0].stats.sampling_rate == 200:
                eventos.loc[(
                    ev_index, pk_index
                    ), 'Warning'].loc[ev_index, pk_index].append(
                        f' Sampling rate is {st[0].stats.sampling_rate}'
                )
                st.decimate(2)
                print(' - Warning! Decimated 2x')

            compo = [tr.stats.component for tr in st]
            cft_array = calculate_cft(st)
            cft_m = cft_max(cft_array)
            print(f' - Componestes: {compo}')

            if len(compo) != 3 or cft_m is None:
                err = f' - Error! len(compo) != 3 ({compo})'
                eventos.loc[
                    (ev_index, pk_index),
                    'Error'
                ].loc[ev_index, pk_index].append(err)
                print(err)
                continue

            spectro = []
            find = False
            for c in compo:
                trace = st.select(component=c)[0]
                s_rate = trace.stats.sampling_rate
                nb_pts = int(WINDOW_LENGTH * s_rate)

                fft_list = []
                time_used = []
                start = trace.stats.starttime
                END = trace.stats.endtime

                while start + WINDOW_LENGTH <= END:
                    tr = trace.slice(starttime=start,
                                     endtime=start + WINDOW_LENGTH)

                    mean_time = tr.stats.starttime + (WINDOW_LENGTH / 2)
                    time_used.append(mean_time - trace.stats.starttime)
                    start += (WINDOW_LENGTH * OVERLAP)

                    fft, _ = get_fft(tr, WINDOW_LENGTH, OVERLAP, nb_pts)

                    fft = np.array(fft)
                    fft_list.append(fft)

                fft_list = np.array(fft_list)
                if fft_list.shape == (237, 50):  # OVERLAP 75% : (237,50)
                    fft_list /= fft_list.max()
                    spectro.append(fft_list)
                    find = True
                else:
                    err = f'fft_list.shape != (237,50) ({fft_list.shape})'
                    eventos.loc[(ev_index, pk_index), 'Error'].loc[ev_index, pk_index].append(err)
                    print(f' - Error! {err}')

            if find is True and len(spectro) == 3:
                spectro = np.array(spectro)
                os.makedirs(f'{spectro_dir}/{ev_index}', exist_ok=True)
                stream_name = (p_path.split('/')[-1]).split('.mseed')[0]
                np.save(f'{spectro_dir}/{ev_index}/{stream_name}.npy', spectro)
                eventos.loc[(ev_index, pk_index), 'Compo'].loc[ev_index, pk_index] = compo
            else:
                err = f'find is {find} and len(spectro) == {len(spectro)}'
                eventos.loc[(ev_index, pk_index), 'Error'].loc[ev_index, pk_index].append(err)
                print(f' - Error! {err}')

    return eventos
