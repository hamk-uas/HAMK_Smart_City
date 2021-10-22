from rnn import MyGRU, MyLSTM, VanillaRNN, RNN
import pandas as pd

# Download the downsampled data frame from csv-file.
raw_data = pd.read_csv(r'C:\Users\iivo210\Desktop\wapice_das\hvac_modeling\data_example.csv')

# Initialize the model with the required parameters.
hvac_model = MyGRU(quant=['Energy_consumption'], seq=4, fut=0, parameters=['Outside_humidity',
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
# Does hyperparameter optimization automatically in pre-defined search space.
hvac_model.fit(X=x_train, y=y_train, epochs=10, max_trials=1)

# Form prediction for energy consumption.
preds = hvac_model.model.predict(x_test)

# Inverse target variables both for measured values and computed predictions.
preds, y_test = hvac_model.inv_target(x_test, preds, y_test, hvac_model.scaler)

# Print the class attributes.
print('Vars:')
print(vars(hvac_model))

# Plot model prediction alongside measured values.
hvac_model.plot_preds(preds, y_test)

# Save the object to folder in the root of the working directory.
hvac_model.save()
