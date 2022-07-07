
"""
Compute the best energy consumption delay with least RMSE among all different delays combination.
Basis to apply  function (14) from "PREDICTIVE OPTIMIZATION OF HEAT DEMAND UTILIZING HEAT STORAGE CAPACITY OF BUILDINGS"
by Petri Hietaharju.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from itertools import combinations


def preprocess_delay( raw_data, dl):
    '''
    Data preprocessing to find energy consumption's delay hour.
    Inputs: Data frames, delay
    Output: Training input data, training target data, testing input data, testing target data, parameters.
    '''
    raw_data = raw_data.copy()
    raw_data.rename(columns={'timestamp': 'Time', }, inplace=True)
    vec = raw_data.iloc[:, 1].values

    datetimes = np.array([[vec, vec], [vec, vec]], dtype='M8[ms]').astype('O')[0, 1]
    raw_data['weekday'] = [t.timetuple().tm_wday for t in datetimes]
    raw_data['hours'] = [t.hour for t in datetimes]

    # drop NAs
    todrop = raw_data[raw_data['Inside_temperature'].isna()].index.values
    raw_data = raw_data.drop(todrop, axis=0, )
    todrop = raw_data[raw_data['Energy_consumption'].isna()].index.values
    raw_data = raw_data.drop(todrop, axis=0, )
    todrop = raw_data[raw_data['Outside_temperature_average'].isna()].index.values
    raw_data = raw_data.drop(todrop, axis=0, )
    raw_data = raw_data.reset_index(drop=True)

    # 'target' is the temperature in next time
    raw_data['target'] = pd.concat([raw_data.loc[1:, 'Inside_temperature'],
                                    pd.Series([raw_data.loc[len(raw_data) - 1, 'Inside_temperature']])],
                                   ignore_index=True)

    # set different hours delay of energy consumption
    raw_data['energy_consumption_1_hours_delay'] = raw_data["Energy_consumption"]
    for i in range(1, 12):
        raw_data['energy_consumption_{}_hours_delay'.format(i + 1)] = pd.concat(
            [raw_data.loc[25 - i:24, 'Energy_consumption'],
             raw_data.loc[:len(raw_data) - i + 1, 'Energy_consumption']], ignore_index=True)

    # set parameters
    parameters = [
        'Outside_temperature_average',
        'Inside_temperature',
    ]
    parameters = parameters + list(dl)

    # Scale all data features to range [0,1]
    scaler = MinMaxScaler()

    df = raw_data.loc[:, parameters + ['target', 'Time']]
    df_train_all = df.loc[:int(len(raw_data) * 0.8), ].copy()
    df_test_all = df.loc[int(len(raw_data) * 0.8):, ].copy()
    df_train = df.loc[:int(len(raw_data) * 0.8), parameters].copy()
    df_test = df.loc[int(len(raw_data) * 0.8):, parameters].copy()

    df_scaler = df.loc[:, parameters].copy()
    scaler.fit(df_scaler)
    df_train = scaler.transform(df_train)
    df_test = scaler.transform(df_test)

    df_train = pd.DataFrame(df_train, columns=parameters)
    df_train['Time'] = df_train_all.loc[:, 'Time'].values

    df_test = pd.DataFrame(df_test, columns=parameters)
    df_test['Time'] = df_test_all.loc[:, 'Time'].values

    X_train = df_train
    y_train = df.loc[:int(len(raw_data) * 0.8), 'target']
    X_test = df_test
    y_test = df.loc[int(len(raw_data) * 0.8):, 'target']

    # Output the shapes of training and testing data.
    # print(f'Shape of training data: {X_train.shape}')
    # print(f'Shape of testing data: {X_test.shape}')

    return X_train, y_train, X_test, y_test, parameters


def build_lr(X_train, y_train, X_test, y_test):
    """
    Linear regression model for prediction
    Input: Training input data, training target data, testing input data, testing target data
    Output: Rmse between prediction and target on test data
    """
    linreg = LinearRegression()

    model = linreg.fit(X_train, y_train)
    # print("coef: ",model.coef_)
    # print("intercept: ", model.intercept_)
    y_pred = model.predict(X_test)
    sum_mean = 0
    for i in range(len(y_pred)):
        sum_mean += (y_pred[i] - y_test.values[i]) ** 2
    sum_erro = np.sqrt(sum_mean / len(y_pred))
    # calculate RMSE
    print("RMSE :", sum_erro)
    return sum_erro

def delay_comb(candidates,n=1):
    """
    Different combination of delay hours
    """
    comb = combinations(candidates, n)
    comb = list(comb)
    print(comb)
    return comb


def tuning(delay_cand,iter):
    """
    Best combination of delay hours
    Input: combination of delays, maximum length of combinations.
    Output: least RMSE, best combination of delay hours with least RMSE
    """
    best_rmse = 9999
    best_delay = ()
    for i in range(iter):
        for delay in delay_comb(delay_cand, i + 1):
            X_train, y_train, X_test, y_test,para= preprocess_delay(raw_data, delay)
            rmse = build_lr(X_train.loc[:,para], y_train, X_test.loc[:,para], y_test)
            if rmse < best_rmse:
                best_rmse = rmse
                best_delay = delay
    return best_rmse, best_delay


#main
# Download the downsampled data frame from csv-file.
raw_data = pd.read_csv(r'3452_building_data_0827.csv')

delay_cand = []
for i in range(12):
    delay_cand.append('energy_consumption_{}_hours_delay'.format(i+1))

# set how many energy consumption delay you want(2,3,4)
num = 2

best_rmse, best_delay = tuning(delay_cand,num)
print('best_rmse',best_rmse)
print('best_delay',best_delay)








