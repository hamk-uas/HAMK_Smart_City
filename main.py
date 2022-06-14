from rnn import MyGRU, MyLSTM, VanillaRNN, RNN
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np

# Uncomment to disable GPU:
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Download the downsampled data frame from csv-file.
raw_data = pd.read_csv(r'data_example_offsets_new.csv')

# Initialize the model with the required parameters.
hvac_model = MyGRU(y_parameters=['Energy_consumption'], seq=12, fut=0, x_parameters=[
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
])

# Scale, split, and sequence the downsampled data frame.
#x_train, y_train, x_test, y_test = hvac_model.preprocess(raw_data, True)

df_train, x_train, y_train, df_val, x_val, y_val = hvac_model.preprocess(raw_data)
       
# Hyperparameter tuning, uncomment these three lines to run it
#print("Cross-validation hyperparameter tuning")
#hvac_model.tune_hyperparameters(x=x_train, y=y_train, epochs=1000, max_trials=15)
#hvac_model.save()

# If everything went OK, find the result folder and append "_hyperparameter_tuning" to the folder name.

# Model training using full training data and best hyperparameters found earlier
hvac_model.load(r'GRU_Energy_consumption_2022-06-14_hyperparameter_tuning')
print("Train model")
hvac_model.retrain(x_train, y_train, x_val, y_val)
hvac_model.save()

# If everything went OK, find the result folder and append "_trained" to the folder name.

################ Iivo's old commented-out code for reference:

# Save the object to folder in the root of the working directory. Uncomment row below, if you have the model trained already.
#hvac_model.load(r'C:\Users\iivo210\Documents\HAMK_Smart_City\GRU_Inside_temperature_2021-11-05')

# Calculating prediction intervals
#rounds = 12     # Number of data instances to calculate prediction intervals to.

#for i in range(rounds):
    
    # Calculating prediction percentiles and saving them to a csv file.
    #hvac_model.prediction_interval(x_train, y_train, x_test[i])     # NB! The process is computationally intensive.

# Making test predictions with the RNN model.
#preds = hvac_model.model.predict(x_train)

# Loading prediction intervals from disk.
#low, up = hvac_model.load_intervals(r'C:\Users\iivo210\Documents\HAMK_Smart_City\GRU_Energy_consumption_2021-10-29\pred_ints.csv')

# Inverse target variables both for measured values and computed predictions.
#preds, y_test = hvac_model.inv_target(x_train, preds, y_train)
#low, up = hvac_model.inv_target(x_test, low, up)

# Plot model prediction alongside measured values.
# Add lower and upper intervals as arguments to plot them.
#hvac_model.plot_preds(preds, y_test)