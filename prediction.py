

import glob
import os 
import csv

import tensorflow as tf
import numpy as np
from numpy import moveaxis



def pred(model_dir, spectro_dir, output_dir) : 
    """Event class prediction
        model_dir : str, path of the input trained model 
        spectro_dir : str, path of the input spectrograms 
        output_dir : str, path of the output file
        
    """

    csvPr_sta = open(os.path.join(output_dir,'prediction_station_level.csv'), 'w')          
    predict_sta = csv.writer(csvPr_sta, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    predict_sta.writerow(['file_name', 
                                 'station',
                                 'prob_nat',
                                 'prob_ant',
                                 'pred',
                                 'nature',
                                     ])  

    
    csvPr_net = open(os.path.join(output_dir,'prediction_network_level.csv'), 'w')          
    predict_net = csv.writer(csvPr_net, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    predict_net.writerow(['event', 
                                 'prob_nat',
                                 'prob_ant',
                                 'pred',
                                 'nature',
                                     ])  

    model = tf.keras.models.load_model(model_dir)

    events = glob.glob(f'{spectro_dir}/*')

    print('Number of events :', len(events))

    nb_evt = 0
    for evt in events : 
        nb_evt +=1
        print('*****************')
        print('EVENT', nb_evt , '/', len(events))
        time = evt.split('/')[2]
        pred_nat = 0
        pred_ant = 0

        list_spect = glob.glob(f'{spectro_dir}/{time}/*')
        print('Number of station :', len(list_spect))
        nb_st = 0
        for spect in list_spect :
            nb_st +=1
            print('Station', nb_st , '/', len(list_spect))
            file_name = (spect.split('/')[3]).split('.npy')[0]
            station = file_name.split('_')[1]
            spect_file = np.load(f'{spect}', allow_pickle=True)
            spect_file = [np.array(spect_file)]
            
            x = moveaxis(spect_file, 1, 3)

            model_output = model.predict(x).round(3) 
            pred = np.argmax(model_output, axis=1)

            if pred == 0 : 
                pred_final = 'Natural'
            if pred == 1 : 
                pred_final = 'Anthropogenic'

            predict_sta.writerow([file_name, 
                                         station,
                                         model_output[0][0], 
                                         model_output[0][1],
                                         pred[0],
                                         pred_final, 
                                         ]) 

            pred_nat+=model_output[0][0]
            pred_ant+=model_output[0][1]
        
        pred_total = [pred_nat, pred_ant]
        pred_total = [(float(i)/sum(pred_total)).round(3) for i in pred_total]
        pred_event = np.argmax(pred_total) 
        if pred_event == 0 : 
            pred_final = 'Natural'
        if pred_event == 1 : 
            pred_final = 'Anthropogenic'   

        predict_net.writerow([time, 
                                pred_total[0], 
                                pred_total[1],
                                pred_event,
                                pred_final, 
                                ]) 


def valid(model_dir, spectro_dir, output_dir, event_label) : 
    """Event class validation
        model_dir : str, input trained model 
        spectro_dir : str, path of the input spectrograms 
        output_dir : str, path of the output file
        event_label : list, class of events to validate
    """

    csvPr_sta = open(os.path.join(output_dir,'validation_station_level.csv'), 'w')          
    predict_sta = csv.writer(csvPr_sta, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    predict_sta.writerow(['file_name', 
                                 'station',
                                 'label_cat',
                                 'prob_nat',
                                 'prob_ant',
                                 'pred',
                                 'nature',
                                     ])  

    
    csvPr_net = open(os.path.join(output_dir,'validation_network_level.csv'), 'w')          
    predict_net = csv.writer(csvPr_net, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    predict_net.writerow(['event', 
                                'label_cat',
                                 'prob_nat',
                                 'prob_ant',
                                 'pred',
                                 'nature'])  

    model = tf.keras.models.load_model(model_dir)


    events = glob.glob(f'{spectro_dir}/*')

    print('Number of events :', len(events))
    print(len(event_label))
    nb_evt = 0
    for a in range(len(event_label)) : 
        nb_evt +=1
        print('*****************')
        print('EVENT', nb_evt , '/', len(event_label))
        time = event_label[a][0]
        
        class_ = event_label[a][1]
        #time = evt.split('/')[2]
        pred_nat = 0
        pred_ant = 0

        list_spect = glob.glob(f'{spectro_dir}/{time}/*')
        print('Number of station :', len(list_spect))
        nb_st = 0
        for spect in list_spect :
            nb_st +=1
            print('Station', nb_st , '/', len(list_spect), end = "\r")
            file_name = (spect.split('/')[3]).split('.npy')[0]
            
            station = file_name.split('_')[1]
            spect_file = np.load(f'{spect}', allow_pickle=True)
            spect_file = [np.array(spect_file)]
            
            x = moveaxis(spect_file, 1, 3)

            model_output = model.predict(x).round(3) 
            pred = np.argmax(model_output, axis=1)

            if pred == 0 : 
                pred_final = 'Natural'
            if pred == 1 : 
                pred_final = 'Anthropogenic'

            predict_sta.writerow([file_name, 
                                         station,
                                         class_,
                                         model_output[0][0], 
                                         model_output[0][1],
                                         pred[0],
                                         pred_final, 
                                         ]) 

            pred_nat+=model_output[0][0]
            pred_ant+=model_output[0][1]
        
        pred_total = [pred_nat, pred_ant]
        pred_total = [(float(i)/sum(pred_total)).round(3) for i in pred_total]
        pred_event = np.argmax(pred_total) 
        if pred_event == 0 : 
            pred_final = 'Natural'
        if pred_event == 1 : 
            pred_final = 'Anthropogenic'   

        predict_net.writerow([time, 
                                class_,
                                pred_total[0], 
                                pred_total[1],
                                pred_event,
                                pred_final, 
                                ]) 


            
            