import obspy as op
from obspy.core import read
import matplotlib.mlab as mlab

import numpy as np
from math import *  
import glob
import os

from obspy.signal.invsim import cosine_taper

def fft_taper(data: np.ndarray) -> np.ndarray:
    """Taper signal
    """
    data *= cosine_taper(npts=len(data),
                         p=0.2)

    return data


def get_fft(trace, WINDOW_LENGTH, OVERLAP, nb_pts) : 
    """Fourier Transform function

        trace : op.Stream()
        WINDOW_LENGTH : int, length (in sec) of the sliding windows
        OVERLAP : float, length of the overlap between each shift
        np_pts : int, number of points in a window (sampling_rate * WINDOWS_LENGTH)
        
    """
    s_rate = trace.stats.sampling_rate
    
    #np_pts = 512
    
    nb_pts = int(WINDOW_LENGTH * s_rate)
    nb_overlap = int(OVERLAP * nb_pts)
    window = fft_taper
    window = window(np.ones(nb_pts, trace.data.dtype))
    
    result = mlab.stride_windows(trace.data, nb_pts, nb_overlap, axis=0)
    result = mlab.detrend(result, mlab.detrend_linear, axis=0)
    result = result * window.reshape((-1, 1))
    numFreqs = nb_pts//2 +1
    result = np.fft.fft(result, n=nb_pts, axis=0)[:numFreqs, :]
    #print(len(result[0]))
    freqs = np.fft.fftfreq(nb_pts, 1/s_rate)[:numFreqs]
    freqs[-1] *= -1
    # Discard the first element (offset)
    result = result[1:]
    freqs = freqs[1:]

    result = np.abs(result)/trace.data.size
    
    result = result.ravel()
    
    return result, freqs



def spectro_extract_valid(data_dir, events_list) : 
    """Spectrogramm extraction for the valid mode
    
        data_dir : str, path of the input mseed
        events_list : list, list of all the events to be validate with their labels
       
    """
    WINDOW_LENGTH = 1
    OVERLAP = (1-0.75)  
    
    print('Number of events :', len(events_list))
    nb_evt = 0
    for a in range(len(events_list)) :  
        nb_evt +=1
        print('*****************')
        print('EVENT', nb_evt , '/', len(events_list))
        time = events_list[a][0]
        
        if not os.path.exists(f'./spectro_demo/{time}'):
            os.makedirs(f'./spectro_demo/{time}')

        list_stream = glob.glob(f'{data_dir}/{time}/*')
        
        print('Number of stream :', len(list_stream))
        nb_st = 0
        for stream in list_stream : 
            nb_st +=1
            print('Stream', nb_st , '/', len(list_stream), end = "\r")
            st = op.read(stream,  dtype=np.dtype(float))
            stream_name = (stream.split('/')[3]).split('.mseed')[0]

            st.detrend('demean')
            st.taper(0.05)
            st = st.filter('highpass', freq=2, corners=4, zerophase=True)

            if st[0].stats.sampling_rate == 200 :
                st.decimate(2)

            compo = []
            
            for tr in st : 
                compo.append(tr.stats.component)
                if len(compo) != 3 : 
                    continue
            spectro = []
            find = False
            for c in compo : 

                trace = st.select(component=c)[0]
                s_rate = trace.stats.sampling_rate
                nb_pts = int(WINDOW_LENGTH * s_rate)

                nyquist_f = trace.stats.sampling_rate / 2

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

                    fft, freqs = get_fft(tr, WINDOW_LENGTH, OVERLAP, nb_pts)

                    fft = np.array(fft)
                    fft_list.append(fft)

                fft_list = np.array(fft_list)
                if fft_list.shape == (237, 50) : #OVERLAP 75% : (237,50)
                    fft_list /= fft_list.max()
                    spectro.append(fft_list)
                    find = True
        
            if find == True and len(spectro)==3 : 
                spectro = np.array(spectro)    
                np.save(f'./spectro_demo/{time}/{stream_name}.npy', spectro)


def spectro_extract_pred(data_dir) : 
    """Spectrogramm extraction for the prediction mode
    
        data_dir : str, path of the input mseed
        
    """
    WINDOW_LENGTH = 1
    OVERLAP = (1-0.75)  
    events = glob.glob(f'{data_dir}/*')
    
    print('Number of events :', len(events))
    nb_evt = 0
    for a in range(len(events)) : 
        nb_evt +=1
        print('*****************')
        print('EVENT', nb_evt , '/', len(events))
        time = events[a].split('/')[2]
        print(time)
        
        if not os.path.exists(f'./spectro_demo/{time}'):
            os.makedirs(f'./spectro_demo/{time}')

        list_stream = glob.glob(f'{data_dir}/{time}/*')
        
        print('Number of stream :', len(list_stream))
        nb_st = 0
        for stream in list_stream : 
            nb_st +=1
            print('Stream', nb_st , '/', len(list_stream), end = "\r")
            st = op.read(stream,  dtype=np.dtype(float))
            
            stream_name = (stream.split('/')[3]).split('.mseed')[0]

            st.detrend('demean')
            st.taper(0.05)
            st = st.filter('highpass', freq=2, corners=4, zerophase=True)

            if st[0].stats.sampling_rate == 200 :
                st.decimate(2)

            compo = []
            
            for tr in st : 
                compo.append(tr.stats.component)
                if len(compo) != 3 : 
                    continue
            spectro = []
            find = False
            for c in compo : 

                trace = st.select(component=c)[0]
                s_rate = trace.stats.sampling_rate
                nb_pts = int(WINDOW_LENGTH * s_rate)

                nyquist_f = trace.stats.sampling_rate / 2

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

                    fft, freqs = get_fft(tr, WINDOW_LENGTH, OVERLAP, nb_pts)

                    fft = np.array(fft)
                    fft_list.append(fft)

                fft_list = np.array(fft_list)
                if fft_list.shape == (237, 50) : #OVERLAP 75% : (237,50)
                    fft_list /= fft_list.max()
                    spectro.append(fft_list)
                    find = True
        
            if find == True and len(spectro)==3 : 
                spectro = np.array(spectro)    
                np.save(f'./spectro_demo/{time}/{stream_name}.npy', spectro)
