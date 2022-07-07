"""
Linear model to predict inside temperature for different prediction intervals.
Basis to apply  function (14) from "PREDICTIVE OPTIMIZATION OF HEAT DEMAND UTILIZING HEAT STORAGE CAPACITY OF BUILDINGS"
by Petri Hietaharju.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

def preprocess_Tampere(raw_data):
    '''
    data preprocessing for Tampere data.
    Inputs: Data frames
    Output: Training input data, training target data, testing input data, testing target data, max inside temperature,
    min inside temperature, parameters
    '''
    raw_data = raw_data.copy()
    raw_data.rename(columns={'timestamp': 'Time',}, inplace=True)
    raw_data.loc[:, 'Time'] = pd.to_datetime(raw_data.loc[:, 'Time'], format='%Y-%m-%d %H:%M:%S')
    vec = raw_data.iloc[:, 1].values

    datetimes = np.array([[vec, vec], [vec, vec]], dtype='M8[ms]').astype('O')[0, 1]

    raw_data['weekday'] = [t.timetuple().tm_wday for t in datetimes]
    raw_data['hours'] = [t.hour for t in datetimes]

    raw_data['hours_sin'] = np.sin(2 * np.pi * raw_data['hours'] / 24.0)
    raw_data['hours_cos'] = np.cos(2 * np.pi * raw_data['hours'] / 24.0)
    raw_data['weekday_sin'] = np.sin(2 * np.pi * raw_data['weekday'] / 7)
    raw_data['weekday_cos'] = np.cos(2 * np.pi * raw_data['weekday'] / 7)

    todrop = raw_data[raw_data['Inside_temperature'].isna()].index.values


    raw_data = raw_data.drop(todrop, axis=0,)
    todrop = raw_data[raw_data['Energy_consumption'].isna()].index.values

    raw_data = raw_data.drop(todrop, axis=0,)
    todrop = raw_data[raw_data['Outside_temperature_average'].isna()].index.values

    raw_data = raw_data.drop(todrop, axis=0,)
    raw_data=raw_data.reset_index(drop=True)



    #'target' is the temperature in next time
    raw_data['target']= pd.concat([raw_data.loc[1:,'Inside_temperature'],pd.Series([raw_data.loc[len(raw_data)-1,'Inside_temperature']])], ignore_index=True)
    raw_data['energy_consumption_1_hours_delay'] = raw_data["Energy_consumption"]



    for i in range(1,12):
            raw_data['energy_consumption_{}_hours_delay'.format(i+1)] = pd.concat([raw_data.loc[25-i:24,'Energy_consumption'], raw_data.loc[:len(raw_data)-i+1,'Energy_consumption']], ignore_index=True)

    # parameters coming from the best delay hours of energy consumption
    parameters = [
        'Outside_temperature_average',
        'energy_consumption_5_hours_delay',
        'energy_consumption_9_hours_delay',
        'energy_consumption_10_hours_delay',
        'energy_consumption_12_hours_delay',
        'Inside_temperature',
                  ]


    # Scale all data features to range [0,1]
    scaler = MinMaxScaler()


    df= raw_data.loc[:,parameters+['target','Time']]
    df_train_all = df.loc[:int(len(raw_data) * 0.8),].copy()
    df_test_all = df.loc[int(len(raw_data) * 0.8):,].copy()
    df_train = df.loc[:int(len(raw_data) * 0.8),parameters].copy()
    df_test = df.loc[int(len(raw_data) * 0.8):,parameters].copy()

    df_scaler=df.loc[:,parameters].copy()
    max_=df_scaler['Inside_temperature'].max()
    min_=df_scaler['Inside_temperature'].min()
    scaler.fit(df_scaler)
    df_train = scaler.transform(df_train)
    df_test = scaler.transform(df_test)
    print("max:{},min:{}".format(max_,min_))


    df_train=pd.DataFrame(df_train,columns=parameters)
    df_train['Time'] = df_train_all.loc[:, 'Time'].values

    df_test=pd.DataFrame(df_test,columns=parameters)
    df_test['Time'] = df_test_all.loc[:, 'Time'].values


    X_train = df_train
    y_train = df.loc[:int(len(raw_data) * 0.8),'target']
    X_test = df_test
    y_test = df.loc[int(len(raw_data) * 0.8):,'target']

    # Output the shapes of training and testing data.
    print(f'Shape of training data: {X_train.shape}')
    print(f'Shape of testing data: {X_test.shape}')

    return X_train, y_train, X_test, y_test,max_,min_,parameters



def build_lr(X_train, y_train):
    """
    Linear regression model for prediction
    Input: Training input data, training target data,
    Output: Linear regression model
    """
    linreg = LinearRegression()

    model = linreg.fit(X_train, y_train)
    # print("coef: ",model.coef_)
    # print("intercept: ", model.intercept_)
    return linreg


def prediction(model, x0):
    """
    compute prediction on single data point x0
    Input:  model, input data row
    Output: prediction on the f
    """

    model = model
    x0=x0.values.reshape(1,-1)

    y_pred = model.predict(x0)
    return  y_pred

def plot_preds(lr, X_test, y_test, fl):
    """
    plot
    Input: linear model, testing input data, testing target data, prediction interval.

    """
    plt.figure()

    #Multiple horizon plot
    plt.title("Different Periods Prediction of Inside Temperature")
    plt.plot(range(len(y_test)), y_test, 'r', label="Measured")

    for f in fl:
        pred = []
        for i in range(len(X_test)):

            y_pred = prediction(lr, X_test.loc[i, :])
            pred.append(y_pred[0])
            # y_scale = (y_pred[0]-min)/(max-min)
            y_scale = (y_pred[0] - min_) / (max_ - min_)

            if i < len(X_test) - 1:
                if i % f == f - 1:
                    continue
                else:
                    X_test.loc[i + 1, 'Inside_temperature'] = y_scale

        pred = np.array(pred)

        # calculate RMSE and MAPE of only predicted ones
        RMSE = np.sqrt(np.sum(np.square(y_test - pred)) / int(len(y_test) * (f - 1) / f))
        MAPE = np.sum(np.abs((pred - y_test) / y_test)) / int(len(y_test) * (f - 1) / f)

        ##calculate RMSE and MAPE of measured ones and predicted ones
        # RMSE = np.sqrt(np.sum(np.square(y_test-pred))/len(y_test))
        # MAPE =np.sum(np.abs((pred-y_test)/y_test))/len(y_test)

        print("RMSE: {}, MAPE: {} for {} hours.".format(RMSE, MAPE, f))

        plt.plot(range(len(pred)), pred, label="Per {} hours prediction".format(f))
    # # simple plot title
    # plt.title("{} Hours Prediction of Inside Temperature\nRMSE: {:.5f}, MAPE: {:.5f}".format(f, RMSE, MAPE))

    plt.legend(loc="upper right")
    plt.xlabel("Date")
    plt.ylabel('Temperature')
    plt.grid()
    plt.show()




def plot_preds_time(lr, X_test, y_test, fl,para,max_,min_):
    """
    plot
    x-axis is date
    Input: linear model, testing input data, testing target data, prediction interval.
    """
    X_test_time = X_test.loc[:,'Time']
    X_test = X_test.loc[:,para]
    plt.figure()

    #Multiple horizon plot
    plt.title("Different Periods Prediction of Inside Temperature")

    plt.plot(X_test_time, y_test, 'r', label="Measured")

    for f in fl:
        pred = []
        for i in range(len(X_test)):

            y_pred = prediction(lr, X_test.loc[i, :])
            pred.append(y_pred[0])
            y_scale = (y_pred[0] - min_) / (max_ - min_)

            if i < len(X_test) - 1:
                if i % f == f - 1:
                    continue
                else:
                    X_test.loc[i + 1, 'Inside_temperature'] = y_scale

        pred = np.array(pred)

        # calculate RMSE and MAPE of only predicted ones
        RMSE = np.sqrt(np.sum(np.square(y_test - pred)) / int(len(y_test) * (f - 1) / f))
        MAPE = np.sum(np.abs((pred - y_test) / y_test)) / int(len(y_test) * (f - 1) / f)

        ##calculate RMSE and MAPE of measured ones and predicted ones
        # RMSE = np.sqrt(np.sum(np.square(y_test-pred))/len(y_test))
        # MAPE =np.sum(np.abs((pred-y_test)/y_test))/len(y_test)

        print("RMSE: {}, MAPE: {} for {} hours.".format(RMSE, MAPE, f))

        plt.plot(X_test_time, pred, label="Per {} hours prediction".format(f))
    # # simple plot title
    # plt.title("{} Hours Prediction of Inside Temperature\nRMSE: {:.5f}, MAPE: {:.5f}".format(f, RMSE, MAPE))

    plt.legend(loc="upper right")
    plt.xlabel("Time series")
    plt.ylabel('Temperature')
    plt.grid()
    plt.show()

#main
raw_data = pd.read_csv(r'3452_building_data_0827.csv')
X_train, y_train, X_test, y_test,max_,min_,para = preprocess_Tampere(raw_data)
lr = build_lr(X_train.loc[:,para], y_train)
f=[4,8,12,24,48]

# plot_preds(lr,X_test.loc[:,para], y_test,f)

#show date in x-axis
plot_preds_time(lr,X_test, y_test,f,para,max_,min_)