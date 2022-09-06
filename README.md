#  New CNN based tool to discriminate anthropogenic from natural low magnitude seismic events



## Related Paper 

C, Hourcade; M, Bonnin; E, Beucler (2022) "New CNN based tool to discriminate anthropogenic from natural low magnitude seismic events" Submitted - Journal Geophysical International

## Dataset 

To apply the algorithm, we need a folder architecture: 
 * mseed_demo
   * 2022004T134407
        * FR_CHLF_2022004T134407.mseed
        * FR_GARF_2022004T134407.mseed
        * FR_GNEF_2022004T134407.mseed
        * FR_VERF_2022004T134407.mseed

Each mseed file corresponds to the raw 3 component recordings of 60 sec. 

## Trained model

Located in directory: model/model_2021354T1554.h5


## Prediction 

The algorithm supports the mseed data format. 

-  mseed format 
```
 run run.py --mode pred --data_dir ./mseed_demo --output_dir ./output_demo --spectro_dir ./spectro_demo
```
<<<<<<< HEAD

Output files are automatically saved in "output_demo". 
The algorithm produces two output files: "prediction_network_level.csv" and "prediction_station_level.csv". 

-  prediction_station_level.csv

This output corresponds to the prediction of each mseed file and therefore for each station. 

| file_name     | station      | prob_nat   | prob_ant   | pred   | nature   
| ------------- | ------------- | --------    |--------    |--------    |--------    |
| FR_CHLF_2022004T134407         | CHLF         | 0.962   |0.038    | 0   | Natural   |
| FR_GARF_2022004T134407         | GARF         | 0.982   |0.018    | 0   | Natural   |
| FR_GNEF_2022004T134407         | GNEF         | 0.914   |0.086    | 0   | Natural   |
| FR_VERF_2022004T134407         | VERF         | 0.985   |0.015    | 0   | Natural   |

-  prediction_network_level.csv

This output corresponds to the prediction of each event. We sum the probabilities for each class for all stations of a given
event and then average them. This gives us an event-based classification.

| event      | prob_nat   | prob_ant   | pred   | nature   
| ------------- | ------------- | --------    |--------    |--------    |
| 2022003T041502         | 0.988           | 0.012   | 0 |  Natural   |
| 2022004T111745          | 0.005   |0.995    | 1   | Anthropogenic   |
| 2022004T111040         | 0.005   |0.995  | 1   | Anthropogenic   |
| 2022004T105235        | 0.011  |0.989 | 1   | Anthropogenic   |



## Validation 

This mode can be used if the label is known. As input you need a csv file with the associated label for each event.
The algorithm supports the mseed data format. 

-  mseed format 
```
 run run.py --mode valid --csv_file ./demo_file.csv --data_dir ./mseed_demo --output_dir ./output_demo --spectro_dir ./spectro_demo
```

Output files are automatically saved in "output_demo". 
The algorithm produces two output files: "prediction_network_level.csv" and "prediction_station_level.csv". 

-  prediction_station_level.csv

This output corresponds to the prediction of each mseed file and therefore for each station. 

| file_name     | station  |   label_cat | prob_nat   | prob_ant   | pred   | nature   |
| ------------- | ------------- | --------    |--------    |--------    |--------    |  --------    |
| FR_CHLF_2022004T134407         | CHLF    | 0     | 0.962   |0.038    | 0   | Natural   |
| FR_GARF_2022004T134407         | GARF    | 0      | 0.982   |0.018    | 0   | Natural   |
| FR_GNEF_2022004T134407         | GNEF    | 0      | 0.914   |0.086    | 0   | Natural   |
| FR_VERF_2022004T134407         | VERF    | 0      | 0.985   |0.015    | 0   | Natural   |

-  prediction_network_level.csv

This output corresponds to the prediction of each event. We sum the probabilities for each class for all stations of a given
event and then average them. This gives us an event-based classification.

| event     | label_cat | prob_nat   | prob_ant   | pred   | nature   |
| ------------- | ------------- | --------    |--------    |--------    |  --------    |
| 2022003T041502    | 0      | 0.988           | 0.012   | 0 |  Natural   |
| 2022004T111745    | 1      | 0.005   |0.995    | 1   | Anthropogenic   |
| 2022004T111040    | 1     | 0.005   |0.995  | 1   | Anthropogenic   |
| 2022004T105235    | 1    | 0.011  |0.989 | 1   | Anthropogenic   |

Output files are automatically saved in "output_demo". 
The algorithm produces two output files: "prediction_network_level.csv" and "prediction_station_level.csv". 

-  prediction_station_level.csv

This output corresponds to the prediction of each mseed file and therefore for each station. 

| file_name     | station      | prob_nat   | prob_ant   | pred   | nature   
| ------------- | ------------- | --------    |--------    |--------    |--------    |
| FR_CHLF_2022004T134407         | CHLF         | 0.962   |0.038    | 0   | Natural   |
| FR_GARF_2022004T134407         | GARF         | 0.982   |0.018    | 0   | Natural   |
| FR_GNEF_2022004T134407         | GNEF         | 0.914   |0.086    | 0   | Natural   |
| FR_VERF_2022004T134407         | VERF         | 0.985   |0.015    | 0   | Natural   |

-  prediction_network_level.csv

This output corresponds to the prediction of each event. We sum the probabilities for each class for all stations of a given
event and then average them. This gives us an event-based classification.

| event      | prob_nat   | prob_ant   | pred   | nature   
| ------------- | ------------- | --------    |--------    |--------    |
| 2022003T041502         | 0.988           | 0.012   | 0 |  Natural   |
| 2022004T111745          | 0.005   |0.995    | 1   | Anthropogenic   |
| 2022004T111040         | 0.005   |0.995  | 1   | Anthropogenic   |
| 2022004T105235        | 0.011  |0.989 | 1   | Anthropogenic   |



## Validation 

This mode can be used if the label is known. As input you need a csv file with the associated label for each event.
The algorithm supports the mseed data format. 

-  mseed format 
```
 run run.py --mode valid --csv_file ./demo_file.csv --data_dir ./mseed_demo --output_dir ./output_demo --spectro_dir ./spectro_demo
```

Output files are automatically saved in "output_demo". 
The algorithm produces two output files: "prediction_network_level.csv" and "prediction_station_level.csv". 

-  prediction_station_level.csv

This output corresponds to the prediction of each mseed file and therefore for each station. 

| file_name     | station  |   label_cat | prob_nat   | prob_ant   | pred   | nature   |
| ------------- | ------------- | --------    |--------    |--------    |--------    |  --------    |
| FR_CHLF_2022004T134407         | CHLF    | 0     | 0.962   |0.038    | 0   | Natural   |
| FR_GARF_2022004T134407         | GARF    | 0      | 0.982   |0.018    | 0   | Natural   |
| FR_GNEF_2022004T134407         | GNEF    | 0      | 0.914   |0.086    | 0   | Natural   |
| FR_VERF_2022004T134407         | VERF    | 0      | 0.985   |0.015    | 0   | Natural   |

-  prediction_network_level.csv

This output corresponds to the prediction of each event. We sum the probabilities for each class for all stations of a given
event and then average them. This gives us an event-based classification.

| event     | label_cat | prob_nat   | prob_ant   | pred   | nature   |
| ------------- | ------------- | --------    |--------    |--------    |  --------    |
| 2022003T041502    | 0      | 0.988           | 0.012   | 0 |  Natural   |
| 2022004T111745    | 1      | 0.005   |0.995    | 1   | Anthropogenic   |
| 2022004T111040    | 1     | 0.005   |0.995  | 1   | Anthropogenic   |
| 2022004T105235    | 1    | 0.011  |0.989 | 1   | Anthropogenic   |

