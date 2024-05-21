# coding: utf-8

import glob
import os

import matplotlib.mlab as mlab
import numpy as np

import obspy as op
from obspy.signal.invsim import cosine_taper

import pandas as pd


def fft_taper(data: np.ndarray) -> np.ndarray:
    """
    Cosine taper for computation of FFT.

    :type data: numpy.ndarray().
    :param data: Input data. Vector that contains the amplitude of a seismic
        trace.

    :return: The tapered data window.
    """

    return data * cosine_taper(npts=data.size, p=0.2)


def get_fft(
        trace: op.core.trace.Trace,
        WINDOW_LENGTH: int,
        OVERLAP: float,
        nb_pts: int) -> tuple:
    """
    Compute the Fourier Transform of a seismic trace.
    :param trace: obspy.Trace() object that contains the seismogram to be
        analyzed.
    :param WINDOW_LENGTH: Length (in second) of the sliding windows.
    :param OVERLAP: Percentage of overlap between each shift.
    :param np_pts: Number of points in a window (sampling_rate *
        WINDOWS_LENGTH)
    :return: 2 numpy.ndarray that contains the frequency vector and the
        amplitude spectrum of the input signal.
    """
    s_rate = trace.stats.sampling_rate

    nb_pts = int(WINDOW_LENGTH * s_rate)
    nb_overlap = int(OVERLAP * nb_pts)
    window = fft_taper
    window = window(np.ones(nb_pts, trace.data.dtype))

    # --
    result = mlab.stride_windows(trace.data, nb_pts, nb_overlap, axis=0)
    result = mlab.detrend(result, mlab.detrend_linear, axis=0)
    # --
    result = result * window.reshape((-1, 1))
    numFreqs = nb_pts // 2 + 1
    result = np.fft.fft(result, n=nb_pts, axis=0)[:numFreqs, :]

    freqs = np.fft.fftfreq(nb_pts, 1 / s_rate)[:numFreqs]
    freqs[-1] *= -1
    # Discard the first element (offset)
    result = result[1:]
    freqs = freqs[1:]

    result = np.abs(result) / trace.data.size

    result = result.ravel()

    return result, freqs


def spectro_extract(
        mseed_dir: str,
        spectro_dir: str,
        eventos: pd.Dataframe) -> None:
    """
        Compute the spectrograms that will be used for the validation.
        The matrices are saved as NumPy objects.

        :type mseed_dir: str
        :param mseed_dir: Absolute path to the input MSEED signal.
        :type spectro_dir: str
        :param spectro_dir: Absolute path where to save the output
            spectrograms.
        :type events_list: list
        :events_list: List of all the labelled events to validate.
    """
    WINDOW_LENGTH = 1
    OVERLAP = (1 - 0.75)

    print(f'Number of events: {eventos.shape[0]}')
    nb_evt = 0
    for a in range(len(events)):
        nb_evt += 1
        print('*****************')
        print(f'EVENT {nb_evt} / {len(events)}')

        if events[a].size == 1:
            time = events[a]
        else:
            time = events[a][0]

        os.makedirs(f'{spectro_dir}/{time}', exist_ok=True)

        list_stream = glob.glob(f'{mseed_dir}/{time}/*')

        print(f'Number of streams: {len(list_stream)}')
        nb_st = 0
        for stream in list_stream:
            nb_st += 1

            print(f'Stream {nb_st} / {len(list_stream)}', end="\r")
            st = op.read(stream, dtype=float)
            stream_name = (stream.split('/')[-1]).split('.mseed')[0]

            st.detrend('demean')
            st.taper(0.05)
            st = st.filter('highpass', freq=2, corners=4, zerophase=True)

            if st[0].stats.sampling_rate == 200:
                st.decimate(2)

            compo = []

            for tr in st:
                compo.append(tr.stats.component)
                if len(compo) != 3:
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

            if find is True and len(spectro) == 3:
                spectro = np.array(spectro)
                np.save(f'{spectro_dir}/{time}/{stream_name}.npy', spectro)
