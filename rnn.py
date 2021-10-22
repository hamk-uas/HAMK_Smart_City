import pandas as pd
import numpy as np
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

class RNN:
    '''
    Parent class for RNN models.
    '''
    def __init__(self, quant=None, seq=4, fut=0, parameters=None):
        '''
        All parameters for class objects are defined here, child classes don't have __init__ methods
        Inputs: target quantities as list, sequence length as int, future period as int, input parameters as a list.
        '''
        self.quant = quant
        self.seq = seq
        self.fut = fut
        self.parameters = parameters
        self.date = date.today()    # For bookkeeping purposes
        self.model = None           # For storage of a model
        self.scaler = None          # For storage of feature scaler
        self.name = None            # Defined after training
        
    def preprocess(self, raw_data):
        '''
        Function for preprocessing downsampled data for sequence modeling.
        Inputs: Downsampled data frame with desired parameters defined in class attribute list in headers
        Output: Training input data, training target data, testing input data, testing target data, sklearn scaler object for inverse transformations
        '''
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
        
        # Extend parameter list by quantity for picking data
        self.parameters.extend(self.quant)
        
        # Split the data to training and testing sets
        raw_data = raw_data[self.parameters].copy()
        df_train = raw_data[int(len(raw_data)*0.2):].copy()
        df_val = raw_data[:int(len(raw_data)*0.2)].copy()
        
        # Delete the quantity from parameter list to preserve the original inputs
        self.parameters = [x for x in self.parameters if x not in self.quant]
        
        # Scale all data features to range [0,1]
        self.scaler = MinMaxScaler()
        df_train = self.scaler.fit_transform(df_train)
        df_val = self.scaler.transform(df_val)
        
        # Next generate a list which will hold all of the sequences for training data
        sequences_train = []
        sequences_val = []
        prev_days_train = deque(maxlen=self.seq)  # Placeholder for the sequences
        prev_days_val = deque(maxlen=self.seq)
        l_quant = len(self.quant)
        
        for count, row in enumerate(pd.DataFrame(df_train).values):
            prev_days_train.append([val for val in row[:-l_quant]]) # store everything but the target values

            if (len(prev_days_train) == self.seq):  # This checks that our sequences are of the correct length and target value is at full hour
                if (any(pd.isna(pd.DataFrame(df_train).values[count-1][-l_quant:]))): # Test for 30 min data interval because of energy data gaps
                    continue
                try:
                    sequences_train.append([np.array(prev_days_train), pd.DataFrame(df_train).values[count+1][-l_quant:]])
                except IndexError:
                    break
            
        for count, row in enumerate(pd.DataFrame(df_val).values):
            prev_days_val.append([val for val in row[:-l_quant]]) # store everything but the target values

            if (len(prev_days_val) == self.seq):  # This checks that our sequences are of the correct length and target value is at full hour
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
        
        print(f'Shape of training data: {X_train.shape}')
        print(f'Shape of testing data: {X_val.shape}')
        
        return X_train, y_train, X_val, y_val
        
    def inv_target(self, X, preds, y_val, scaler):
        '''
        Method for inverting the scaling target variable
        Inputs: 3-dimensional data matrix used to train (or validate) the model, predictions obtained using the model,
                validation target vector and pre-fitted sklearn scaler. 
                Note: the X tensor is more of a placeholder in this function used only for getting the dimensions correct.
        Output: Inversely transformed predictions and validation vectors
        '''
        
        # The number of quantities used in optimization
        N = np.array(preds).shape[1]
        
        preds = np.concatenate((X[:len(preds),-1], np.array(preds).reshape(len(preds), N)), axis=1) # Reshape is necessary as there are issues with dimensions
        y_val = np.concatenate((X[:len(preds),-1], np.array(y_val[:len(preds)]).reshape(len(preds), N)), axis=1)
        
        preds = scaler.inverse_transform(preds)[:,-N:]
        y_val = scaler.inverse_transform(y_val)[:,-N:]
        
        return preds, y_val
        
    def plot_preds(self, preds, y_val):
        '''
        Producing plots of predictions with the measured values as time series.
        Inputs: predicted and measured values as numpy arrays
        '''
        plt.figure()
        
        plt.plot(preds, color='navy', label='Predicted')
        plt.plot(y_val, color='darkorange', label='Measured', marker='*')
        plt.legend()
        plt.grid()
        plt.title(f'Predictions for {self.quant[0]} with {self.name}.')
        
        plt.show()
        
    def save(self, path=rf'{os.getcwd()}'):
        '''
        Method for saving the model, scaler, and other attributes to compatible forms.
        Uses same folder as subclasses fit-method to save the information.
        Input: Desired path for saving the information.
        '''
    
        # Define the folder which the results are saved to
        new_fold_path = rf'{path}/{self.name}_{self.quant[0]}_{str(self.date)}'
        if not os.path.exists(new_fold_path):    # Test whether the directory already exists
            os.makedirs(new_fold_path)
            print(f'Folder created on path: {new_fold_path}')
        else:
            print(f'Savings results to {new_fold_path}')
        
        # Save model to folder
        self.model.save(rf'{new_fold_path}/model.h5')
        print('Model saved.')
        
        # Save scaler to folder
        dump(self.scaler, rf'{new_fold_path}/scaler.joblib')
        print('Scaler saved.')
        
        # Save all other variables to json format to folder
        other_vars = {'name': self.name, 'quant': self.quant, 'seq': self.seq, 'fut': self.fut, 'parameters': self.parameters, 'date': str(self.date)}
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
        self.quant = var_dict["quant"]
        self.seq = var_dict["seq"]
        self.fut = var_dict["fut"]
        self.parameters = var_dict["parameters"]
        self.date = var_dict["date"]
        
        print('Other variables loaded.')
        

class CVTuner(kt.engine.tuner.Tuner):
    '''
    Class used for customizing Keras Tuner for cross-validation purposes. Inherits Tuner baseclass.
    By default, 5-fold CV is implemented.
    '''
    
    def run_trial(self, trial, x, y, batch_size=32, epochs=1, patience=20):
        cv = KFold(5)
        val_losses = []
        for train_indices, test_indices in cv.split(x):
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
        hp_units = hp.Int('units', min_value=self.units[0], max_value=self.units[1], step=10)
        hp_layers = hp.Int('layers', min_value=self.layers[0], max_value=self.layers[1])
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
    def fit(self, X, y, epochs, max_trials, units=[10, 100], act=['tanh', 'relu'], layers=[1, 2], lr=[0.1, 0.01, 0.001]):
        '''
        Fitting method performing hyperparameter optimization. Bayesian Optimization is used for finding correct
        direction in search space, while 5-fold cross-validation is used for measuring predictive performance of
        a model. Saves the model object and the name to class attributes.
        Inputs: Preprocessed input and target data as numpy arrays, maximum epochs for training as int, model compositions to be tested as int,
                hyperparameter search space with fitting default values.
        '''
        tuner = CVTuner(hypermodel=RNN_HyperModel(mtype='GRU', input_shape=(X.shape[1], X.shape[2]), units=[10,100],
                            act=act, layers=layers, lr=lr),
                            oracle=kt.oracles.BayesianOptimization(objective='val_loss', max_trials=max_trials),
                            directory=os.getcwd(),
                            project_name=f'VanillaRNN_{self.quant[0]}_{str(date.today())}', overwrite=True)
        
        tuner.search(X, y, epochs=epochs)
        
        print(tuner.results_summary())
        
        best = tuner.get_best_models(num_models=1)[0]
        self.name = f'VanillaRNN'
        self.model = best
        
class MyGRU(RNN):
    '''
    Gated Recurrent Unit variant of RNN. Inherits all attributes and methods from parent class.
    '''
    def fit(self, X, y, epochs, max_trials, units=[10, 100], act=['tanh'], layers=[1, 2], lr=[0.1, 0.01, 0.001]):
        '''
        Fitting method performing hyperparameter optimization. Bayesian Optimization is used for finding correct
        direction in search space, while 5-fold cross-validation is used for measuring predictive performance of
        a model. Saves the model object and the name to class attributes.
        Inputs: Preprocessed input and target data as numpy arrays, maximum epochs for training as int, model compositions to be tested as int,
                hyperparameter search space with fitting default values.
        '''
        tuner = CVTuner(hypermodel=RNN_HyperModel(mtype='GRU', input_shape=(X.shape[1], X.shape[2]), units=[10,100],
                            act=act, layers=layers, lr=lr),
                            oracle=kt.oracles.BayesianOptimization(objective='val_loss', max_trials=max_trials),
                            directory=os.getcwd(),
                            project_name=f'GRU_{self.quant[0]}_{str(date.today())}', overwrite=True)
        
        tuner.search(X, y, epochs=epochs)
        
        print(tuner.results_summary())
        
        best = tuner.get_best_models(num_models=1)[0]
        self.name = f'GRU'
        self.model = best
        
class MyLSTM(RNN):
    '''
    Long Short Term Memory variant of RNN. Inherits all attributes and methods from parent class.
    '''
    def fit(self, X, y, epochs, max_trials, units=[10, 100], act=['tanh'], layers=[1, 2], lr=[0.1, 0.01, 0.001]):
        '''
        Fitting method performing hyperparameter optimization. Bayesian Optimization is used for finding correct
        direction in search space, while 5-fold cross-validation is used for measuring predictive performance of
        a model. Saves the model object and the name to class attributes.
        Inputs: Preprocessed input and target data as numpy arrays, maximum epochs for training as int, model compositions to be tested as int,
                hyperparameter search space with fitting default values.
        '''
        tuner = CVTuner(hypermodel=RNN_HyperModel(mtype='GRU', input_shape=(X.shape[1], X.shape[2]), units=[10,100],
                            act=act, layers=layers, lr=lr),
                            oracle=kt.oracles.BayesianOptimization(objective='val_loss', max_trials=max_trials),
                            directory=os.getcwd(),
                            project_name=f'LSTM_{self.quant[0]}_{str(date.today())}', overwrite=True)
        
        tuner.search(X, y, epochs=epochs)
        
        print(tuner.results_summary())
        
        best = tuner.get_best_models(num_models=1)[0]
        self.name = f'LSTM'
        self.model = best