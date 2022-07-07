"""
2022-06-07
Linear model to predict inside temperature for different prediction horizons based on Tampere data.
Energy optimization to reduce the inside temperature during the time the building is occupied(8.00-16.00).
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")


def preprocess_Tampere(raw_data,start_hour,ending_hour,pct=1.0):
    '''
    data preprocessing for Tampere data.
    Inputs: Data frames, starting and ending time of cutting down, percent of cutting down
    Output: Training input data, training target data, testing input data, testing target data, max inside temperature,
    min inside temperature,parameters, outdoor temperature, heating after cutting down.
    '''
    raw_data = raw_data.copy()
    raw_data.rename(columns={'timestamp': 'Time',}, inplace=True)

    raw_data.loc[:, 'Time'] = pd.to_datetime(raw_data.loc[:, 'Time'], format='%Y-%m-%d %H:%M:%S')
    vec = raw_data.loc[:, 'Time'].values
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

    # cut down energy comsumption by a certain percentage
    df_cut = raw_data.copy()
    if start_hour > ending_hour:
        ending_hour = 24 + ending_hour
    rg = range(start_hour, ending_hour + 1)
    for i in rg:
        i = i % 24
        df_cut["Energy_consumption"][df_cut['hours'].values == i] = df_cut["Energy_consumption"][df_cut['hours'].values == i] * pct

    # generate 12 hour delays of energy consumption
    df_cut['energy_consumption_1_hours_delay'] = df_cut["Energy_consumption"]
    for i in range(1,12):
            raw_data['energy_consumption_{}_hours_delay'.format(i+1)] = pd.concat([raw_data.loc[25-i:24,'Energy_consumption'], raw_data.loc[:len(raw_data)-i+1,'Energy_consumption']], ignore_index=True)
            df_cut['energy_consumption_{}_hours_delay'.format(i + 1)] = pd.concat([df_cut.loc[25 - i:24, 'Energy_consumption'], df_cut.loc[:len(raw_data) - i + 1, 'Energy_consumption']], ignore_index=True)


    outtemp = df_cut.loc[int(len(raw_data) * 0.8):,'Outside_temperature_average'].copy()
    heat = df_cut.loc[int(len(raw_data) * 0.8):, 'energy_consumption_1_hours_delay'].copy()

    # the energy_consumption_x_hours_delay come from results of best_delay.py
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

    df_cut = df_cut.copy()

    df = raw_data.copy()
    df_train_all = df.loc[:int(len(raw_data) * 0.8),].copy()
    df_test_all = df.loc[int(len(raw_data) * 0.8):,].copy()
    df_cut_test_all = df_cut.loc[int(len(raw_data) * 0.8):, ].copy()
    df_train = df.loc[:int(len(raw_data) * 0.8),parameters].copy()
    df_test = df.loc[int(len(raw_data) * 0.8):,parameters].copy()
    df_cut_test = df_cut.loc[int(len(raw_data) * 0.8):, parameters].copy()

    df_scaler=df.loc[:,parameters].copy()
    max_pre=df_scaler['Inside_temperature'].max()
    min_pre=df_scaler['Inside_temperature'].min()
    scaler.fit(df_scaler)
    df_train = scaler.transform(df_train)
    df_test = scaler.transform(df_test)
    df_cut_test = scaler.transform(df_cut_test)
    # print("max:{},min:{}".format(max_pre,min_pre))


    df_train=pd.DataFrame(df_train,columns=parameters)
    df_test=pd.DataFrame(df_test,columns=parameters)
    df_cut_test = pd.DataFrame(df_cut_test, columns=parameters)
    df_train['hours'] = df_train_all.loc[:,'hours'].values
    df_test['hours'] = df_test_all.loc[:,'hours'].values
    df_cut_test['hours']= df_cut_test_all.loc[:, 'hours'].values
    df_cut_test['Time'] = df_cut_test_all.loc[:, 'Time'].values
    df_cut_test['Energy_consumption'] = df_cut_test_all.loc[:, 'Energy_consumption'].values


    X_train = df_train.copy()
    y_train = df.loc[:int(len(raw_data) * 0.8),'target']
    X_test = df_cut_test.copy()
    y_test = df.loc[int(len(raw_data) * 0.8):,'target']

    # # Output the shapes of training and testing data.
    # print(f'Shape of training data: {X_train.shape}')
    # print(f'Shape of testing data: {X_test.shape}')

    return X_train, y_train, X_test, y_test,max_pre,min_pre,parameters,outtemp,heat

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

    x0=x0.values.reshape(1,-1)
    y_pred = model.predict(x0)
    return  y_pred


def morning_cut(raw_data,f, setting_temp = 20,start_hour = 7,end_hour=16,  pct=0.3):
    """
    calculate end hour of cutting meanwhile the inside temperature is above setting one during the required period.
    """
    ending = range(start_hour+1,end_hour+1)

    for ending_hour in ending:
        X_train, y_train, X_test, y_test, max_pre,min_pre, parameters, outtemp, heat = preprocess_Tampere(raw_data, start_hour, ending_hour,pct)
        lr = build_lr(X_train[parameters], y_train)

        pred = []
        for i in range(len(X_test)):

            y_pred = prediction(lr, X_test.loc[i, parameters])
            # pred.append(y_pred[0])
            y_scale = (y_pred[0] - min_pre) / (max_pre - min_pre)
            if i < len(X_test) - 1:
                if i % f != f - 1:
                    X_test.loc[i + 1, 'Inside_temperature'] = y_scale

            if y_pred[0] >= setting_temp:

                continue
            else:
                print("The longest {:.1f}% cutting period from {} is to {} with {} hours prediction horizon".format((1-pct)*100,start_hour, ending_hour % 24-1, f))
                return ending_hour-1

        print('All temperatures are over {} ℃ from {} to {} with {} hours prediction horizon'.format(setting_temp,start_hour,ending_hour % 24,f))

    # print("{}%".format((1-pct)*100))
    return ending_hour





def plot_cut(raw_data,start_hour,end, setting_temp,f, ):
    """
    plot figure 17 from petri's thesis.
    """

    X_train, y_train, X_test_ori, y_test, max, min, parameters, outtemp, heat = preprocess_Tampere(raw_data, start_hour,
                                                                                               start_hour)
    plt.figure()


    plt.subplot(3,1,1)
    plt.plot(X_test_ori['Time'], y_test, 'r', label="Measured")
    plt.subplot(3, 1, 2)
    plt.plot(X_test_ori['Time'], heat, 'r', label="Measured")

    best_cutting = 0
    best_pred = []
    best_pct = 0
    best_ending = 0
    best_heating = []


    for j in np.arange(0,1,0.01):
        ending_hour = morning_cut(raw_data, f, setting_temp, start_hour,end, j)
        X_train, y_train, X_test, y_test,max_pre,min_pre,parameters,outtemp,heat = preprocess_Tampere( raw_data,start_hour, ending_hour, j)
        lr = build_lr(X_train[parameters], y_train)
        cutting = np.sum(X_test_ori['Energy_consumption']-X_test['Energy_consumption'])
        pred = []

        for i in range(len(X_test)):
            y_pred = prediction(lr, X_test.loc[i, parameters])
            pred.append(y_pred[0])
            y_scale = (y_pred[0] - min_pre) / (max_pre - min_pre)

            if i < len(X_test) - 1:
                if i % f == f - 1:
                    continue
                else:
                    X_test.loc[i + 1, 'Inside_temperature'] = y_scale

        pred = np.array(pred)
        if best_cutting < cutting:
            best_cutting = cutting
            best_pred = pred
            best_ending = ending_hour
            best_heating = heat
            best_pct = j


    print(best_cutting)
    # calculate RMSE and MAPE of only predicted ones
    RMSE = np.sqrt(np.sum(np.square(y_test - best_pred)) / int(len(y_test) * (f - 1) / f))
    MAPE = np.sum(np.abs((best_pred - y_test) / y_test)) / int(len(y_test) * (f - 1) / f)
    print("RMSE: {}, MAPE: {} for {} hours with {:.1f}% cutting.".format(RMSE, MAPE, f,(1 - best_pct) * 100))
    plt.subplot(3, 1, 1)
    plt.plot(X_test_ori['Time'], best_pred, label="{:.1f}% Cutting(setting: {})".format((1-best_pct)*100,setting_temp))
    plt.subplot(3, 1, 2)
    plt.plot(X_test_ori['Time'], best_heating, label="{:.1f}% Cutting".format((1 - best_pct) * 100))

    plt.subplot(3, 1, 1)
    plt.legend(loc="best")
    plt.ylabel('Indoor Temperature(℃)')
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.legend(loc="best")
    plt.ylabel('Heating power(kwh)')
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.ylabel('Outdoor Temperature(℃)')
    plt.plot(X_test_ori['Time'], outtemp )
    plt.grid()
    plt.suptitle("Predictions of Inside Temperature with Different Energy Cutting from {}:00 to {}:00 saving {:.2f} kWh ".format(start_hour,best_ending % 24,best_cutting))
    plt.show()


#main
raw_data = pd.read_csv(r'3452_building_data_0827.csv')

# meature inside temperature every f(int,>1) hours
f=48
# set target temperature
setting_temp = 21.5
# set starting and ending time of the target period
start_hour = 8
ending_hour = 16
# plot
plot_cut(raw_data,start_hour,ending_hour, setting_temp,f,)



