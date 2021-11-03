from rnn import MyGRU, MyLSTM, VanillaRNN, RNN
import pandas as pd

# Download the downsampled data frame from csv-file.
raw_data = pd.read_csv(r'C:\Users\iivo210\Documents\HAMK_Smart_City\data_example.csv')

# Initialize the model with the required parameters.
hvac_model = MyGRU(quant=['Energy_consumption'], seq=12, fut=0, parameters=['Outside_humidity',
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
              'Radiator_network_2_temperature'])
              
              

# Scale, split, and sequence the downsampled data frame.
x_train, y_train, x_test, y_test = hvac_model.preprocess(raw_data)

# Train the model using custom fit method. 
# Does hyperparameter optimization automatically in pre-defined search space. Comment row below, if you have already trained the model.
#hvac_model.fit(X=x_train, y=y_train, epochs=1000, max_trials=5)

# Save the object to folder in the root of the working directory. Uncomment row below, if you have the model trained already.
hvac_model.load(r'C:\Users\iivo210\Documents\HAMK_Smart_City\GRU_Energy_consumption_2021-10-29')

# Calculating prediction intervals
#rounds = 12     # Number of data instances to calculate prediction intervals to.

#for i in range(rounds):
    
    # Calculating prediction percentiles and saving them to a csv file.
    #hvac_model.prediction_interval(x_train, y_train, x_test[i])     # NB! The process is computationally intensive.

# Making test predictions with the RNN model.
preds = hvac_model.model.predict(x_test)

# Loading prediction intervals from disk.
#low, up = hvac_model.load_intervals(r'C:\Users\iivo210\Documents\HAMK_Smart_City\GRU_Energy_consumption_2021-10-29\pred_ints.csv')

# Inverse target variables both for measured values and computed predictions.
preds, y_test = hvac_model.inv_target(x_test, preds, y_test, hvac_model.scaler)
#low, up = hvac_model.inv_target(x_test, low, up, hvac_model.scaler)

# Plot model prediction alongside measured values.
# Add lower and upper intervals as arguments to plot them.
hvac_model.plot_preds(preds, y_test)