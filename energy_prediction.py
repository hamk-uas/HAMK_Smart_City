"""
Linear model to predict energy consumption.
Basis to apply  function (15)-(17) from "PREDICTIVE OPTIMIZATION OF HEAT DEMAND UTILIZING HEAT STORAGE CAPACITY OF BUILDINGS"
by Petri Hietaharju.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

def preprocess_energy(raw_data):
    '''
    data preprocessing for Tampere data.
    Inputs: Data frames
    Output: Training input data, training target data, testing input data, testing target data, parameters,
     max heating, min heating, parameters
    '''
    raw_data = raw_data.copy()
    raw_data.rename(columns={'timestamp': 'Time',}, inplace=True)
    raw_data.loc[:, 'Time'] = pd.to_datetime(raw_data.loc[:, 'Time'], format='%Y-%m-%d %H:%M:%S')

    vec = raw_data.iloc[:, 0].values
    datetimes = np.array([[vec, vec], [vec, vec]], dtype='M8[ms]').astype('O')[0, 1]
    raw_data['weekday'] = [t.timetuple().tm_wday for t in datetimes]
    raw_data['hours'] = [t.hour for t in datetimes]

    raw_data['hours_sin'] = np.sin(2 * np.pi * raw_data['hours'] / 24.0)
    raw_data['hours_cos'] = np.cos(2 * np.pi * raw_data['hours'] / 24.0)
    raw_data['weekday_sin'] = np.sin(2 * np.pi * raw_data['weekday'] / 7)
    raw_data['weekday_cos'] = np.cos(2 * np.pi * raw_data['weekday'] / 7)

    todrop = raw_data[raw_data['Inside_temperature'].isna()].index.values
    raw_data = raw_data.drop(todrop, axis=0,)


    todrop = raw_data[raw_data['Outside_temperature_average'].isna()].index.values
    raw_data = raw_data.drop(todrop, axis=0, )

    todrop = raw_data[raw_data['Energy_consumption'].isna()].index.values
    raw_data = raw_data.drop(todrop, axis=0, )
    raw_data.reset_index(drop=True, inplace=True)

    raw_data['heating'] = raw_data['Energy_consumption']

    # 'last_heating' is the energy comsumption in the last hour
    raw_data['last_heating'] = pd.concat(
        [pd.Series([raw_data.loc[0, "heating"]]), raw_data.loc[0:len(raw_data) - 1, "heating"]], ignore_index=True)


    # set inside temperature = 21.5
    raw_data['Inside_temperature'] = 21.5

    raw_data['diff'] = raw_data['Inside_temperature'] - raw_data['Outside_temperature_average']

    parameters = ['last_heating',
                  'diff',
                  # 'Outside_temperature_average',
                  # 'Inside_temperature',
                  ]

    df = raw_data.loc[:, parameters + ["Time","hours", "heating"]]


    df_train = df.loc[:int(len(raw_data) * 0.8), parameters + ["Time","hours"]].copy()
    df_test = df.loc[int(len(raw_data) * 0.8):, parameters + ["Time","hours"]].copy()
    lh = df_test['last_heating']
    # Scale all data features to range [0,1]
    max_lh = df['last_heating'].max()
    min_lh = df['last_heating'].min()
    max_diff = df['diff'].max()
    min_diff = df['diff'].min()

    # max_in=df['Inside_temperature'].max()
    # min_in=df['Inside_temperature'].min()
    #
    # max_out=df['Outside_temperature_average'].max()
    # min_out=df['Outside_temperature_average'].min()

    df_train['last_heating'] = (df_train['last_heating'] - min_lh) / (max_lh - min_lh)
    df_test['last_heating'] = (df_test['last_heating'] - min_lh) / (max_lh - min_lh)
    df_train['diff'] = (df_train['diff'] - min_diff) / (max_diff - min_diff)
    df_test['diff'] = (df_test['diff'] - min_diff) / (max_diff - min_diff)
    # df_train['Inside_temperature'] = (df_train['Inside_temperature']-min_in)/(max_in-min_in)
    # df_test['Inside_temperature'] = (df_test['Inside_temperature'] - min_in) / (max_in - min_in)

    # df_train['Outside_temperature_average'] = (df_train['Outside_temperature_average']-min_out)/(max_out-min_out)
    # df_test['Outside_temperature_average'] = (df_test['Outside_temperature_average'] - min_out) / (max_out- min_out)

    X_train = df_train.loc[:, parameters + ["Time","hours"]]
    y_train = df.loc[:int(len(raw_data) * 0.8), 'heating']
    X_test = df_test.loc[:, parameters + ["Time","hours"]]
    y_test = df.loc[int(len(raw_data) * 0.8):, 'heating']


    # Output the shapes of training and testing data.
    # print(f'Shape of training data: {X_train.shape}')
    # print(f'Shape of testing data: {X_test.shape}')

    return X_train, y_train, X_test, y_test, parameters, max_lh, min_lh


def build_lr(X_train, y_train):
    """
    Linear regression model for prediction
    Input: Training input data, training target data, parameters
    Output: Linear regression model
    """
    linreg = LinearRegression()

    model = linreg.fit(X_train, y_train)

    return model,

def residual(model,X_train, y_train,para):
    """
    residual model: (16) from 2.3.2 in Petri's thesis
    """
    y_train_pred = model.predict(X_train[para])
    res = []
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    for i in range(24):
        idx = X_train.loc[X_train['hours'] == i, :].index
        res.append(sum(-y_train_pred[idx]+y_train[idx])/len(idx))
    return res

def prediction_energy(model, x0,para,res):
    """
    compute prediction on single data point x0
    Input:  model, input data row,residual,parameters
    Output: prediction
    """

    x0_para = x0[para]

    x0_para=x0_para.values.reshape(1,-1)

    y_pred = model.predict(x0_para)+res[int(x0['hours'])]

    return  y_pred

def plot_energy(lr, X_test, y_test,para,res,f,max_lh,min_lh):

    X_test_time = X_test.loc[:,'Time']
    X_test = X_test.loc[:,para+['hours']]

    plt.figure(figsize=(18,12))

    plt.title("Energy Model")
    plt.plot(X_test_time, y_test, 'r', label="Measurement")

    for f in f:
        pred = []
        scl = []
        for i in range(len(X_test)):


            y_pred =prediction_energy(lr, X_test.loc[X_test.index[i], :],para,res)
            pred.append(y_pred[0])
            y_scale = (y_pred[0] - min_lh) / (max_lh - min_lh)
            scl.append(y_scale)
            if i < len(X_test) - 1:
                if i % f == f - 1:

                    continue
                else:

                    X_test.loc[X_test.index[i] + 1, 'last_heating'] = y_scale

        RMSE = np.sqrt(np.sum(np.square(y_test-pred))/len(y_test))
        MAPE =np.sum(np.abs((pred-y_test)/y_test))/len(y_test)

        print("RMSE: {}, MAPE: {} for {} hours.".format(RMSE, MAPE, f))

        plt.plot(X_test_time, pred, label="Per {} hours prediction".format(f))

    plt.legend(loc="upper right")
    plt.xlabel("Date")
    plt.ylabel('Energy')
    plt.grid()
    plt.show()

raw_data = pd.read_csv(r'3452_building_data_0827.csv')
# prediction interval
f=[4,8]
X_train, y_train, X_test, y_test,para,max_lh,min_lh = preprocess_energy(raw_data)
lr,= build_lr(X_train[para], y_train)
res = residual(lr,X_train, y_train,para)
plot_energy(lr, X_test, y_test,para,res,f,max_lh,min_lh)
