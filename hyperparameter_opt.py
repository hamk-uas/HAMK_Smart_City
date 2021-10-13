import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from collections import deque
import matplotlib.pyplot as plt
import kerastuner as kt
from datetime import datetime
from preprocess import preprocess

# DEFINE DATA AND OUTPUT PATH
output_path = r''
data_path = r''

# Parameter definitions
EPOCHS = 1000               # how many times are we passing through the data at most, early stopping may interrupt this process.
QUANTITY = ['Energy_consumption']       # Predicted quantity
SEQ_LEN = 4                 # How long is the historical sequence for each prediction
FUTURE = 0                  # How long into the future each prediction is made
MAX_TRIALS = 15             # How many iterations on model structure are run
PATIENCE = 20               # How manu consecutive epochs the early stopping has patience, is used for every cv iterations separately

# Read the data from csv
train_data = pd.read_csv(f'{data_path}')

# Preprocessing the data
X_train, y_train, X_val, y_val, scaler = preprocess(train_data, quant=QUANTITY, seq=SEQ_LEN, fut=FUTURE)

# Model builder for tuner
def model_builder(hp):

    model = Sequential()
    
    # Define the hyperparameter search space
    hp_units = hp.Int('units', min_value=10, max_value=120, step=10, default=20)
    hp_layers = hp.Int('layers', min_value=1, max_value=3, default=1)
    hp_act = hp.Fixed('activation function', value='tanh')
    hp_lr = hp.Choice('learning rate', values=[0.001, 0.01, 0.1])
    
    # Additional GRU layers (1-2)
    for i in range(hp_layers):
    
        if i == 0 and max(range(hp_layers)) == 0:
            model.add(GRU(units=hp_units, activation=hp_act, input_shape=(X_train.shape[1], X_train.shape[2])))
        elif i == 0:
            model.add(GRU(units=hp_units, activation=hp_act, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
            model.add(BatchNormalization())
        elif i < max(range(hp_layers)):
            model.add(GRU(units=hp_units, activation=hp_act, return_sequences=True))
            model.add(BatchNormalization())
        else:
            model.add(GRU(units=hp_units, activation=hp_act))
    
    # Output layer
    model.add(Dense(1))
    
    # Define the learning rate in the optimizer
    opt = Adam(learning_rate=hp_lr)
    
    # Compiling the model
    model.compile(loss='mse', optimizer=opt, metrics=['mae'])
    
    return model
    
# Define cv keras tuner class
class CVTuner(kt.engine.tuner.Tuner):
    def run_trial(self, trial, x, y, batch_size=32, epochs=1):
        cv = KFold(5)
        val_losses = []
        for train_indices, test_indices in cv.split(x):
            x_train, x_test = x[train_indices], x[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            model = self.hypermodel.build(trial.hyperparameters)
            # Define early stopping callback with patience parameter
            stopper = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
            model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_test, y_test), epochs=epochs, callbacks=[stopper])
            val_losses.append(model.evaluate(x_test, y_test))
        self.oracle.update_trial(trial.trial_id, {'val_loss': np.mean(val_losses)})
        self.save_model(trial.trial_id, model)

# Creating keras tuner object
tuner = CVTuner(hypermodel=model_builder, oracle=kt.oracles.BayesianOptimization(objective='val_loss', max_trials=MAX_TRIALS), directory=r'C:\Users\iivo210\Documents\smartcityTakahuhti',
                                project_name=f'{QUANTITY[0]}', overwrite=True)

# Conduct the search
tuner.search(X_train, y_train, epochs=EPOCHS)

# Summary of search results
print(tuner.results_summary())

# Save the best model to disk
today = datetime.today().strftime('%Y-%m-%d')
name = rf'\model_{today}_{QUANTITY[0]}.h5'
best = tuner.get_best_models(num_models=1)[0]
print(output_path + name)
best.save(output_path + name)
