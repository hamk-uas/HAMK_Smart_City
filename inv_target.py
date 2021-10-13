import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def inv_target(X, preds, y_val, scaler):
    '''
    Function for inverting the scaling target variable
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