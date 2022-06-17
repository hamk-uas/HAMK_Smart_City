import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import Variable
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GRU, LSTM, SimpleRNN, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from collections import deque
import matplotlib.pyplot as plt
import kerastuner as kt
from datetime import datetime, date
from joblib import dump, load
import os
import json
import csv
import math

class RNN:
    '''
    Parent class for RNN models.
    '''
    
    def __init__(self, y_parameters=None, seq=4, fut=0, x_parameters=None):
        '''
        All parameters for class objects are defined here, child classes don't have __init__ methods
        Inputs: target quantities as list, sequence length as int, future period as int, input parameters as a list.
        '''
        self.x_parameters = x_parameters
        self.y_parameters = y_parameters
        self.used_parameters = x_parameters + y_parameters
        self.seq = seq
        self.fut = fut
        self.date = date.today()    # For bookkeeping purposes
        self.model = None           # For storage of a model
        self.scaler = None          # For storage of feature scaler
        self.name = None            # Defined after training
        
    def preprocess(self, raw_data, fit_scaler = True):
        
        '''
        Function for preprocessing downsampled data for sequence modeling.
        Inputs: Downsampled data frame with desired parameters defined in class attribute list in headers
        fit_scaler: True to fit the scaler, False to use existing scaler (self.scaler)
        Output: Training input data, training target data, testing input data, testing target data, sklearn scaler object for inverse transformations
        '''
        raw_data.iloc[:,0] = pd.to_datetime(raw_data.iloc[:,0], format='%Y-%m-%d %H:%M:%S%z')
        vec = raw_data.iloc[:,0].values
        datetimes = np.array([[vec, vec], [vec, vec]], dtype = 'M8[ms]').astype('O')[0,1]
        raw_data['weekday'] = [t.timetuple().tm_wday for t in datetimes]
        raw_data['hours'] = [t.hour for t in datetimes]
        
        # Encode time parameters to cyclical features
        raw_data['hours_sin'] = np.sin(2 * np.pi * raw_data['hours']/24)
        raw_data['hours_cos'] = np.cos(2 * np.pi * raw_data['hours']/24)
        raw_data['weekday_sin'] = np.sin(2 * np.pi * raw_data['weekday']/7)
        raw_data['weekday_cos'] = np.cos(2 * np.pi * raw_data['weekday']/7)
        
        # Split the data to training and testing sets        
        df_val = raw_data[:int(len(raw_data)*0.2)].copy()
        df_train = raw_data[int(len(raw_data)*0.2):].copy()

        # Scale all used data features to range [0,1]
        if fit_scaler:
            self.scaler = MinMaxScaler()
            df_train_scaled = pd.DataFrame(self.scaler.fit_transform(df_train[self.used_parameters]), columns = self.used_parameters)
        else:
            df_train_scaled = pd.DataFrame(self.scaler.transform(df_train[self.used_parameters]), columns = self.used_parameters)
        df_val_scaled = pd.DataFrame(self.scaler.transform(df_val[self.used_parameters]), columns = self.used_parameters)
       
        # Convert to sequences
        if self.fut == 0:
            x_train = df_train_scaled[self.x_parameters]              # Full timeseries
            y_train = df_train_scaled[self.y_parameters][self.seq-1:] # Skip seq-1 time points from the beginning
            x_val = df_val_scaled[self.x_parameters]                  # Full timeseries
            y_val = df_val_scaled[self.y_parameters][self.seq-1:]     # Skip seq-1 time points from the beginning
        else:
            x_train = df_train_scaled[self.x_parameters][:-self.fut]           # Skip fut time points from the end
            y_train = df_train_scaled[self.y_parameters][self.seq-1+self.fut:] # Skip seq - 1 + fut time points from the beginning
            x_val = df_val_scaled[self.x_parameters][:-self.fut]               # Skip fut time points from the end
            y_val = df_val_scaled[self.y_parameters][self.seq-1+self.fut:]     # Skip seq - 1 + fut time points from the beginning
            
        x_train = np.squeeze(np.lib.stride_tricks.sliding_window_view(x_train, (self.seq, len(self.x_parameters))), axis = 1)
        y_train = y_train.to_numpy()
        x_val = np.squeeze(np.lib.stride_tricks.sliding_window_view(x_val, (self.seq, len(self.x_parameters))), axis = 1)
        y_val = y_val.to_numpy()
        
        # Output the shapes of training and testing data.
        print(f'Shape of training data: x: {x_train.shape} y: {y_train.shape}')
        print(f'Shape of testing data: x: {x_val.shape} y: {y_val.shape}')
                
        return df_train, x_train, y_train, df_val, x_val, y_val
        
    def get_pred_dates(self, df):
        return df.to_numpy()[self.seq-1+self.fut:, 0]
        
    def inv_target(self, x, preds, y_val):
        '''
        Method for inverting the scaling target variable
        Inputs: 3-dimensional data matrix used to train (or validate) the model, predictions obtained using the model,
                validation target vector and pre-fitted sklearn scaler. 
                Note: the x tensor is more of a placeholder in this function used only for getting the dimensions correct.
        Output: Inversely transformed predictions and validation vectors
        '''
        
        preds = np.concatenate((x[:len(preds),-1], np.array(preds).reshape(len(preds), 1)), axis=1) # Reshape is necessary as there are issues with dimensions
        y_val = np.concatenate((x[:len(preds),-1], np.array(y_val[:len(preds)]).reshape(len(preds), 1)), axis=1)
        
        preds = self.scaler.inverse_transform(preds)[:,-1:]
        y_val = self.scaler.inverse_transform(y_val)[:,-1:]
        
        return preds, y_val
        
    def plot_preds(self, datetimes, preds, y_val, color, label, low=[], up=[], conf=0.9):
        '''
        Producing plots of predictions with the measured values as time series.
        Inputs: predicted and measured values as numpy arrays.
        '''
        
        # Number of instances to plot.
        if len(low) != 0:   # Check whether the list is empty.
            rounds = len(low)
        else:
            rounds = len(preds)
        
        plt.plot(datetimes[:rounds], preds[:rounds], color=color, label=label) #darkorange
        plt.plot(datetimes[:rounds], y_val[:rounds], color='red', label='Measured', marker='.', linestyle="")
        if len(low) != 0:     # Check whether the list is empty.
            plt.fill_between(range(rounds), (preds[:rounds,0])+(low[:,0]), (preds[:rounds,0])+(up[:,0]), color='gray', alpha=0.25, label=f'{round(conf*100)}% prediction interval')
        plt.legend()
        plt.grid()
        plt.title(f'Predictions for {self.y_parameters[0]} with {self.name}.')
        
    def load_intervals(self, int_path, conf=0.9):
        '''
        Method for loading desired prediction intervals for ML forecasts.
        Inputs: path to the prediction interval .csv file, confidence level as float (0.5-0.99)
        '''
        
        # Load the predictions
        with open(int_path) as csvf:
            
            read_fil = csv.reader(csvf)
            percs = list(read_fil)
            
        percs = np.array([obj for obj in percs if obj])
        
        low_ind = round(((1-conf)/2 - 0.01) * 100)
        up_ind = round((conf + (1-conf)/2 - 0.01) * 100)
        
        # Select the desired intervals bounds. Reshape is necessary for following target inversion.
        lower, upper = percs[:,low_ind].reshape(len(percs), 1), percs[:,up_ind].reshape(len(percs), 1)
        
        return lower, upper
        
        #plt.figure()
        #
        #plt.plot(preds, label='Predicted')
        #plt.plot(y_val, label='Measured', marker='*')
        #plt.fill_between(range(len(preds)), (preds)+(percs[:,low_ind]), (preds)+(percs[:,up_ind]), color='gray', alpha=0.25, label=f'{round(100*conf)}% prediction interval')
        #plt.legend()
        #
        #plt.show()
                
    def save(self, path=rf'{os.getcwd()}'):
        '''
        Method for saving the model, scaler, and other attributes to compatible forms.
        Uses same folder as subclasses fit-method to save the information.
        Input: Desired path for saving the information.
        '''
    
        # Define the folder which the results are saved to
        new_fold_path = rf'{path}/{self.name}_{self.y_parameters[0]}_{str(self.date)}'
        if not os.path.exists(new_fold_path):    # Test whether the directory already exists
            os.makedirs(new_fold_path)
            print(f'Folder created on path: {new_fold_path}.')
        else:
            print(f'Savings results to {new_fold_path}.')
        
        # Save model to folder
        self.model.save(rf'{new_fold_path}/model.h5')
        print('Model saved.')
        
        # Save scaler to folder
        dump(self.scaler, rf'{new_fold_path}/scaler.joblib')
        print('Scaler saved.')
        
        # Save all other variables to json format to folder
        other_vars = {'name': self.name, 'y_parameters': self.y_parameters, 'seq': self.seq, 'fut': self.fut, 'x_parameters': self.x_parameters, 'date': str(self.date)}

        with open(rf'{new_fold_path}/vars.json', 'w') as f:
            json.dump(other_vars, f)
        print('Other variables saved.')
        
    def load(self, path):
        '''
        Loads RNN model information saved with .save method from location specified in function call.
        Stores the information by updating class attributes.
        Input: path of the storage directory
        '''
        
        # Load the model to class attribute
        self.model = load_model(rf'{path}/model.h5')
        print('Model loaded.')
        
        # Load the scaler
        self.scaler = load(rf'{path}/scaler.joblib')
        print('Scaler loaded.')
        
        # Load dictionary containing all other variables
        with open(rf'{path}/vars.json', 'r') as f:
            var_dict = json.load(f)
            
        # Place the variables to correct positions
        self.name = var_dict["name"]
        self.y_parameters = var_dict["y_parameters"]
        self.seq = var_dict["seq"]
        self.fut = var_dict["fut"]
        self.x_parameters = var_dict["x_parameters"]
        self.used_parameters = self.x_parameters + self.y_parameters
        self.date = var_dict["date"]
        
        print('Other variables loaded.')
        
    def retrain(self, x, y, x_val, y_val, epochs=1000, best_of_n=5):
        best_model = None
        for i in range(best_of_n):
            self.model = tf.keras.models.clone_model(self.model) # Init weights
            opt = Adam(learning_rate=0.01)
            self.model.compile(loss='mse', optimizer=opt, metrics=['mae'])        
            stopper = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            history = self.model.fit(x, y, batch_size=np.shape(x)[0], validation_data=(x_val, y_val), epochs=epochs, callbacks=[stopper])
            val_loss = history.history["val_loss"][-1]
            if best_model == None or val_loss < best_val_loss:
                best_model = self.model
                best_val_loss = val_loss
        self.model = best_model        
        
    def prediction_interval(self, x_train, y_train, x0, path=rf'{os.getcwd()}'):
        '''
        Compute bootstrap prediction interval around the models prediction on single data point x0.
        Inputs: pre-trained model, training input data, training output data, new input data row, number of rows to save,
                path for model saving.
        Output: Percentiles 0-100 for prediction intervals
        '''
        
        # Define output path for saving the percentile results.
        new_fold_path = rf'{path}/{self.name}_{self.y_parameters[0]}_{str(self.date)}'
        if not os.path.exists(new_fold_path):    # Test whether the directory already exists
            os.makedirs(new_fold_path)
            print(f'Folder created on path: {new_fold_path}.')
        else:
            print(f'Savings prediction intervals to {new_fold_path}.')
        
        # Local copy of the machine learning model. Done dut to weight and bias initialization done in the script.
        model = self.model
        
        # Number of training samples
        n = x_train.shape[0]
        
        # Calculate the next prediction to be output in the end
        pred_x0 = model.predict(np.reshape(x0, (1, x0.shape[0], x0.shape[1])))
        
        # Calculate training residuals
        preds = model.predict(x_train)
        train_res = y_train - preds
        
        # Number of bootstrap samples
        n_boots = np.sqrt(n).astype(int)
        
        # Compute bootstrap predictions and validation residuals
        boot_preds, val_res = np.empty(n_boots), []
        
        for b in range(n_boots):
            
            # Reset model weights, not straightforward with tensorflow Recurrent Neural Networks
            for ix, layer in enumerate(model.layers):
                if hasattr(self.model.layers[ix], 'recurrent_initializer'):
                    weight_initializer = model.layers[ix].kernel_initializer
                    bias_initializer = model.layers[ix].bias_initializer
                    recurr_init = model.layers[ix].recurrent_initializer

                    old_weights, old_biases, old_recurrent = model.layers[ix].get_weights()

                    model.layers[ix].set_weights([
                        weight_initializer(shape=old_weights.shape),
                        bias_initializer(shape=old_biases.shape),
                        recurr_init(shape=old_recurrent.shape)])
                elif hasattr(model.layers[ix], 'kernel_initializer') and hasattr(model.layers[ix], 'bias_initializer'):
                    weight_initializer = model.layers[ix].kernel_initializer
                    bias_initializer = model.layers[ix].bias_initializer
                    
                    old_weights, old_biases = model.layers[ix].get_weights()
                    
                    model.layers[ix].set_weights([
                        weight_initializer(shape=old_weights.shape),
                        bias_initializer(shape=len(old_biases))])

            
            print(f'Starting bootstrap {b+1}/{n_boots}')
            train_idx = np.random.choice(range(n), size=n, replace=True)    # Draw the training indexes with replacement
            val_idx = np.array([idx for idx in range(n) if idx not in train_idx])   # Use the ones left after training as validation data
            
            # Train model with training data, validate with validation data. Early Stopping stops training after validation performance
            # starts to deteriorate.
            model.fit(x_train[train_idx], y_train[train_idx], epochs=100, verbose=0, validation_data=(x_train[val_idx], y_train[val_idx]),
                        callbacks=EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True))
                
            preds_val = model.predict(x_train[val_idx]) # Validation predictions
            
            val_res.append(y_train[val_idx] - preds_val)    # Calculate validation residuals
            boot_preds[b] = model.predict(np.reshape(x0, (1, x0.shape[0], x0.shape[1])))   # Predict with bootstrapped model
            
        boot_preds -= np.mean(boot_preds)   # Center bootstrap predictions
        val_res = np.concatenate(val_res, axis=None)    # Flattening predictions to a single array
        
        # Take percentiles of training and validation residuals to compare
        val_res = np.percentile(val_res, q=np.arange(100))
        train_res = np.percentile(train_res, q=np.arange(100))
        
        # Estimates for the relationship between bias and variance
        no_inf_err = np.mean(np.abs(np.random.permutation(y_train) - np.random.permutation(preds)))
        gener = np.abs(val_res.mean() - train_res.mean())
        no_inf_val = np.abs(no_inf_err - train_res)
        rel_overfitting_rate = np.mean(gener / no_inf_val)
        w = .632 / (1 - .368*rel_overfitting_rate)
        res = (1-w) * train_res + w*val_res
        
        # Construct interval boundaries
        C = np.array([m + o for m in boot_preds for o in res])
        percs = np.percentile(C, q=np.arange(0, 101))
        
        # Saving results to model folder...
        print(f'Saving results to {new_fold_path}.')
        
        # Writing rows to file.
        with open(rf'{new_fold_path}/pred_ints.csv', 'a') as f:
            write = csv.writer(f)
            write.writerow(percs)
        
        print('----------------------------------------------------------------------------------------------')
        
        

class CVTuner(kt.engine.tuner.Tuner):
    '''
    Class used for customizing Keras Tuner for cross-validation purposes. Inherits Tuner baseclass.
    By default, 5-fold CV is implemented.
    '''
    
    def run_trial(self, trial, x, y, batch_size=32, epochs=1, patience=20):
        cv = KFold(5)
        val_losses = []
        for train_indices, test_indices in cv.split(x):
            print("Fold")
            x_train, x_test = x[train_indices], x[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            model = self.hypermodel.build(trial.hyperparameters)
            # Define early stopping callback with patience parameter
            stopper = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
            model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_test, y_test), epochs=epochs, callbacks=[stopper])
            val_losses.append(model.evaluate(x_test, y_test))
        self.oracle.update_trial(trial.trial_id, {'val_loss': np.mean(val_losses)})
        self.save_model(trial.trial_id, model)
        
class RNN_HyperModel(kt.HyperModel):
    '''
    Class for custom implementation of Keras Tuner HyperModel. Two methods: initiation with parameters and formation of the hypermodel.
    Inherits Keras Tuner HyperModel base class. Is used in fit-method of child classes.
    Inputs: model type as string, input data shape as tuple, unit boundaries as list, layer boundaries as list,
            learning rate values as list, suitable activation functions as a list.
    '''

    def __init__(self, mtype, input_shape, units, layers, lr, act):
        self.mtype = mtype
        self.input_shape = input_shape
        self.units = units
        self.layers = layers
        self.lr = lr
        self.act = act

    def build(self, hp):
        
        # Create TensorFlow sequential model
        model = Sequential()
        
        # Define hyperparameter search space
        try:
            hp_units = hp.Int('units', min_value=self.units[0], max_value=self.units[1], step=10)
        except IndexError:
            hp_units = hp.Fixed('units', value=self.units[0])
        try:
            hp_layers = hp.Int('layers', min_value=self.layers[0], max_value=self.layers[1])
        except IndexError:
            hp_layers = hp.Fixed('layers', value=self.layers[0])
        hp_act = hp.Choice('activation function', values=self.act)
        hp_lr = hp.Choice('learning rate', values=self.lr)
        
        # Select correct implementation of layer formation based on the model type.
        if self.mtype == 'SimpleRNN':
        
            for i in range(hp_layers):
                if i == 0 and max(range(hp_layers)) == 0:
                    model.add(SimpleRNN(units=hp_units, activation=hp_act, input_shape=self.input_shape))
                elif i == 0:
                    model.add(SimpleRNN(units=hp_units, activation=hp_act, input_shape=self.input_shape, return_sequences=True))
                    model.add(BatchNormalization())
                elif i < max(range(hp_layers)):
                    model.add(SimpleRNN(units=hp_units, activation=hp_act, return_sequences=True))
                    model.add(BatchNormalization())
                else:
                    model.add(SimpleRNN(units=hp_units, activation=hp_act))
        elif self.mtype == 'GRU':
        
            for i in range(hp_layers):
                if i == 0 and max(range(hp_layers)) == 0:
                    model.add(GRU(units=hp_units, activation=hp_act, input_shape=self.input_shape))
                elif i == 0:
                    model.add(GRU(units=hp_units, activation=hp_act, input_shape=self.input_shape, return_sequences=True))
                    model.add(BatchNormalization())
                elif i < max(range(hp_layers)):
                    model.add(GRU(units=hp_units, activation=hp_act, return_sequences=True))
                    model.add(BatchNormalization())
                else:
                    model.add(GRU(units=hp_units, activation=hp_act))
        elif self.mtype == 'LSTM':
            
            for i in range(hp_layers):
                if i == 0 and max(range(hp_layers)) == 0:
                    model.add(LSTM(units=hp_units, activation=hp_act, input_shape=self.input_shape))
                elif i == 0:
                    model.add(LSTM(units=hp_units, activation=hp_act, input_shape=self.input_shape, return_sequences=True))
                    model.add(BatchNormalization())
                elif i < max(range(hp_layers)):
                    model.add(LSTM(units=hp_units, activation=hp_act, return_sequences=True))
                    model.add(BatchNormalization())
                else:
                    model.add(LSTM(units=hp_units, activation=hp_act)) 
        
        # Add a single output cell with linear activation function.
        model.add(Dense(1))
        
        # Define model optimizer, here Adam is used with learning rate decided with Bayesian Optimization
        opt = Adam(learning_rate=hp_lr)
        
        # Compile the model. Mean Squared Error is used as loss function while Mean Absolute Error is calculated for illustration
        model.compile(loss='mse', optimizer=opt, metrics=['mae'])
        
        return model
        
class VanillaRNN(RNN):
    '''
    Conventional Recurrent Neural Network model.
    '''
    def tune_hyperparameters(self, x, y, epochs, max_trials, units=[10, 100], act=['tanh', 'relu'], layers=[1, 2], lr=[0.1, 0.01, 0.001]):
        '''
        Fitting method performing hyperparameter optimization. Bayesian Optimization is used for finding correct
        direction in search space, while 5-fold cross-validation is used for measuring predictive performance of
        a model. Saves the model object and the name to class attributes.
        Inputs: Preprocessed input and target data as numpy arrays, maximum epochs for training as int, model compositions to be tested as int,
                hyperparameter search space with fitting default values.
        '''
        tuner = CVTuner(hypermodel=RNN_HyperModel(mtype='SimpleRNN', input_shape=(x.shape[1], x.shape[2]), units=[10,100],
                            act=act, layers=layers, lr=lr),
                            oracle=kt.oracles.BayesianOptimization(objective='val_loss', max_trials=max_trials),
                            directory=os.getcwd(),
                            project_name=f'VanillaRNN_{self.y_parameters[0]}_{str(date.today())}', overwrite=True)
        
        tuner.search(x, y, epochs=epochs)
        
        print(tuner.results_summary())
        
        best = tuner.get_best_models(num_models=1)[0]
        self.name = f'VanillaRNN'
        self.model = best
        
class MyGRU(RNN):
    '''
    Gated Recurrent Unit variant of RNN. Inherits all attributes and methods from parent class.
    '''
    def tune_hyperparameters(self, x, y, epochs, max_trials, units=[110], act=['tanh'], layers=[1], lr=[0.01]):
        '''
        Fitting method performing hyperparameter optimization. Bayesian Optimization is used for finding correct
        direction in search space, while 5-fold cross-validation is used for measuring predictive performance of
        a model. Saves the model object and the name to class attributes.
        Inputs: Preprocessed input and target data as numpy arrays, maximum epochs for training as int, model compositions to be tested as int,
                hyperparameter search space with fitting default values.
        '''
        tuner = CVTuner(hypermodel=RNN_HyperModel(mtype='GRU', input_shape=(x.shape[1], x.shape[2]), units=units,
                            act=act, layers=layers, lr=lr),
                            oracle=kt.oracles.BayesianOptimization(objective='val_loss', max_trials=max_trials),
                            directory=os.getcwd(),
                            project_name=f'GRU_{self.y_parameters[0]}_{str(date.today())}', overwrite=True)
        
        tuner.search(x, y, epochs=epochs)
        
        print(tuner.results_summary())
        
        best = tuner.get_best_models(num_models=1)[0]
        self.name = f'GRU'
        self.model = best
        
class MyLSTM(RNN):
    '''
    Long Short Term Memory variant of RNN. Inherits all attributes and methods from parent class.
    '''
    def tune_hyperparameters(self, x, y, epochs, max_trials, units=[10, 100], act=['tanh'], layers=[1, 2], lr=[0.1, 0.01, 0.001]):
        '''
        Fitting method performing hyperparameter optimization. Bayesian Optimization is used for finding correct
        direction in search space, while 5-fold cross-validation is used for measuring predictive performance of
        a model. Saves the model object and the name to class attributes.
        Inputs: Preprocessed input and target data as numpy arrays, maximum epochs for training as int, model compositions to be tested as int,
                hyperparameter search space with fitting default values.
        '''
        tuner = CVTuner(hypermodel=RNN_HyperModel(mtype='LSTM', input_shape=(x.shape[1], x.shape[2]), units=[10,100],
                            act=act, layers=layers, lr=lr),
                            oracle=kt.oracles.BayesianOptimization(objective='val_loss', max_trials=max_trials),
                            directory=os.getcwd(),
                            project_name=f'LSTM_{self.y_parameters[0]}_{str(date.today())}', overwrite=True)
        
        tuner.search(x, y, epochs=epochs)
        
        print(tuner.results_summary())
        
        best = tuner.get_best_models(num_models=1)[0]
        self.name = f'LSTM'
        self.model = best