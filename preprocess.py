import numpy as np
import pandas as pd
from collections import deque
from sklearn.preprocessing import MinMaxScaler

def preprocess(raw_data, quant, seq, fut):
    '''
    Function for preprocessing downsampled data for sequence modeling.
    Inputs: Downsampled data frame, target quantity, sequence length used in modelling, and target distance from last observation in input sequence
    Output: Training input data, training target data, testing input data, testing target data, sklearn scaler object for inverse transformations
    '''
    
    # Define input features
    parameters = [
              'Outside_humidity',
              'Solar_irradiance',
              'CO2_concentration',
              'hours_sin',
              'hours_cos',
              'weekday_sin',
              'weekday_cos',
              'Domestic_water_network_1_primary_valve',
              'Domestic_water_network_2_primary_valve',
              'District_heat_temperature',
              'Outside_temperature_average',
              'Ventilation_network_1_temperature',
              'Ventilation_network_2_temperature',
              'Radiator_network_1_temperature',
              'Radiator_network_2_temperature'
              
    ]
   
    # Concoct temporal variables from datetime column
    raw_data.iloc[:,0] = pd.to_datetime(raw_data.iloc[:,0], format='%Y-%m-%d %H:%M:%S%z')
    vec = raw_data.iloc[:,0].values
    datetimes = np.array([[vec, vec], [vec, vec]], dtype = 'M8[ms]').astype('O')[0,1]
    raw_data['weekday'] = [t.timetuple().tm_wday for t in datetimes]
    raw_data['hours'] = [t.hour for t in datetimes]
    
    # Encode time parameters to cyclical features
    raw_data['hours_sin'] = np.sin(2 * np.pi * raw_data['hours']/24.0)
    raw_data['hours_cos'] = np.cos(2 * np.pi * raw_data['hours']/24.0)
    raw_data['weekday_sin'] = np.sin(2 * np.pi * raw_data['weekday']/7)
    raw_data['weekday_cos'] = np.cos(2 * np.pi * raw_data['weekday']/7)
    
    # Extend parameter list by quantity
    parameters.extend(quant)
    
    # Split the data to training and validation sets
    raw_data = raw_data[parameters].copy()
    df_train = raw_data[int(len(raw_data)*0.2):].copy()
    df_val = raw_data[:int(len(raw_data)*0.2)].copy()
    
    # Scale all data to range [0, 1]
    # First fit the scaler to training data and then use the same scale for validation data
    scaler = MinMaxScaler()
    df_train = scaler.fit_transform(df_train)
    df_val = scaler.transform(df_val)
    
    # Next generate a list which will hold all of the sequences for training data
    sequences_train = []
    sequences_val = []
    prev_days_train = deque(maxlen=seq)  # Placeholder for the sequences
    prev_days_val = deque(maxlen=seq)
    l_quant = len(quant)
    
    for count, row in enumerate(pd.DataFrame(df_train).values):
        prev_days_train.append([val for val in row[:-l_quant]]) # store everything but the target values

        if (len(prev_days_train) == seq):  # This checks that our sequences are of the correct length and target value is at full hour
            if (any(pd.isna(pd.DataFrame(df_train).values[count-1][-l_quant:]))): # Test for 30 min data interval because of energy data gaps
                continue
            try:
                sequences_train.append([np.array(prev_days_train), pd.DataFrame(df_train).values[count+1][-l_quant:]])
            except IndexError:
                break
        
    for count, row in enumerate(pd.DataFrame(df_val).values):
        prev_days_val.append([val for val in row[:-l_quant]]) # store everything but the target values

        if (len(prev_days_val) == seq):  # This checks that our sequences are of the correct length and target value is at full hour
            if (any(pd.isna(pd.DataFrame(df_val).values[count-1][-l_quant:]))): # Test for 30 min data interval because of energy data gaps
                continue
            try:
                sequences_val.append([np.array(prev_days_val), pd.DataFrame(df_val).values[count+1][-l_quant:]])
            except IndexError:
                break
            
    # Iterating through the sequences in order to differentiate X and y
    X_train = []
    y_train = []
    X_val = []
    y_val = []

    for seq, target in sequences_train:
        X_train.append(seq)
        y_train.append(target)
        
    for seq, target in sequences_val:
        X_val.append(seq)
        y_val.append(target)
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
        
    print(f'Shape of training data: X {X_train.shape}, y {y_train.shape}')
    print(f'Shape of testing data: X {X_val.shape}, y {y_val.shape}')
 
    return X_train, y_train, X_val, y_val, scaler