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
hvac_model = MyGRU(y_parameters=['Energy_consumption'], seq=12, fut=1, x_parameters=[
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
df_train, x_train, y_train, df_val, x_val, y_val = hvac_model.preprocess(raw_data)

# Enable the following code blocks one at a time and follow the instructions:

# Hyperparameter tuning
if False:
    print("Cross-validation hyperparameter tuning")
    hvac_model.tune_hyperparameters(x=x_train, y=y_train, epochs=1000, max_trials=15)
    hvac_model.save()

# If everything went OK, find the result folder and append "_hyperparameter_tuning_fut_0" to the folder name.

# Model training using full training data and best hyperparameters found earlier
if False:
    #hvac_model.load(r'GRU_Energy_consumption_2022-06-14_hyperparameter_tuning_fut_0')
    hvac_model.load(r'GRU_Energy_consumption_2022-06-15_hyperparameter_tuning_fut_1')
    print("Train model")
    hvac_model.retrain(x_train, y_train, x_val, y_val)
    hvac_model.save()

# If everything went OK, find the result folder and append "_trained_fut_0" to the folder name.

# Calculate statistics on scenario testing results over bootstrapped model training on input sequences resampled with replacement:

if True:
    hvac_model.load(r'GRU_Energy_consumption_2022-06-15_trained_fut_1')

    raw_data_offsets_removed = raw_data.copy();
    raw_data_offsets_removed['Radiator_network_2_temperature'] -= raw_data['Radiator_network_2_offset']
    raw_data_offsets_removed['Ventilation_network_2_temperature'] -= raw_data['Ventilation_network_2_offset']
    df_train_offsets_removed, x_train_offsets_removed, y_train_offsets_removed, df_val_offsets_removed, x_val_offsets_removed, y_val_offsets_removed = hvac_model.preprocess(raw_data_offsets_removed, False)

    pred_dates_train = hvac_model.get_pred_dates(df_train)
    pred_dates_val = hvac_model.get_pred_dates(df_val)
    bootstrap_samples = 20

    predictedEnergyConsumptionScenarioEnabled = [pred_dates_train.astype('str')]
    predictedEnergyConsumptionScenarioDisabled = [pred_dates_train.astype('str')]
    for i in range(bootstrap_samples):
        train_idx = np.random.choice(range(np.shape(x_train)[0]), size=np.shape(x_train)[0], replace=True) # Draw the training indexes with replacement
        hvac_model.retrain(x_train[train_idx], y_train[train_idx], x_val, y_val)
        preds, y_scenario = hvac_model.inv_target(x_train, hvac_model.model.predict(x_train), y_train)
        preds_offsets_removed, y_offsets_removed = hvac_model.inv_target(x_train_offsets_removed, hvac_model.model.predict(x_train_offsets_removed), y_train_offsets_removed)
        predictedEnergyConsumptionScenarioEnabled.append(preds.flatten().astype('str'))
        predictedEnergyConsumptionScenarioDisabled.append(preds_offsets_removed.flatten().astype('str'))

    predictedEnergyConsumptionScenarioEnabled = np.array(predictedEnergyConsumptionScenarioEnabled).T
    np.savetxt("predictedEnergyConsumptionScenarioEnabled.csv", predictedEnergyConsumptionScenarioEnabled, fmt='%s', delimiter=',')

    predictedEnergyConsumptionScenarioDisabled = np.array(predictedEnergyConsumptionScenarioDisabled).T
    np.savetxt("predictedEnergyConsumptionScenarioDisabled.csv", predictedEnergyConsumptionScenarioDisabled, fmt='%s', delimiter=',')

