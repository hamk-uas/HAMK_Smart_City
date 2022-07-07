"""
Control curve optimization.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from scipy import optimize
# ignore warning
import warnings
warnings.filterwarnings("ignore")


def preprocess(raw_data,):
    '''
    Function for preprocessing data for modeling.
    Inputs: Data frames
    Output: Training input data, training target data, testing input data, testing target data, max inside temperature,
    min inside temperature,max radiator temperature,min radiator temperature,max outside temperature,min outside temperature.
    '''
    raw_data = raw_data.copy()
    raw_data.rename(columns={'timestamp': 'Time', }, inplace=True)
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
    raw_data = raw_data.drop(todrop, axis=0, )
    todrop = raw_data[raw_data['Energy_consumption'].isna()].index.values

    raw_data = raw_data.drop(todrop, axis=0, )
    todrop = raw_data[raw_data['Outside_temperature_average'].isna()].index.values

    raw_data = raw_data.drop(todrop, axis=0, )
    raw_data = raw_data.reset_index(drop=True)


    #'target' is the temperature in next time
    raw_data['target']= pd.concat([raw_data.loc[1:,'Inside_temperature'],pd.Series([raw_data.loc[len(raw_data)-1,'Inside_temperature']])], ignore_index=True)
    raw_data['energy_consumption_1_hours_delay'] = raw_data["Energy_consumption"]

    for i in range(1,12):
            raw_data['energy_consumption_{}_hours_delay'.format(i+1)] = pd.concat([raw_data.loc[25-i:24,"Energy_consumption"], raw_data.loc[:len(raw_data)-i+1,"Energy_consumption"]], ignore_index=True)

    parameters = [
        'Outside_temperature_average',
        'Radiator_network_1_temperature',
        'Inside_temperature',
                  ]


    # Scale all data features to range [0,1]
    scaler = MinMaxScaler()


    df= raw_data.loc[:,parameters+['target']]


    df_train = df.loc[:int(len(raw_data) * 0.8),parameters].copy()
    df_test = df.loc[int(len(raw_data) * 0.8):,parameters].copy()

    df_scaler=df.loc[:,parameters].copy()
    max_in=df_scaler['Inside_temperature'].max()
    min_in=df_scaler['Inside_temperature'].min()
    max_ra=df_scaler['Radiator_network_1_temperature'].max()
    min_ra=df_scaler['Radiator_network_1_temperature'].min()
    max_out=df_scaler['Outside_temperature_average'].max()
    min_out=df_scaler['Outside_temperature_average'].min()
    scaler.fit(df_scaler)
    df_train = scaler.transform(df_train)
    df_test = scaler.transform(df_test)



    df_train=pd.DataFrame(df_train,columns=parameters)
    df_test=pd.DataFrame(df_test,columns=parameters)


    X_train = df_train.loc[:,parameters]
    y_train = df.loc[:int(len(raw_data) * 0.8),'target']
    X_test = df_test.loc[:,parameters]
    y_test = df.loc[int(len(raw_data) * 0.8):,'target']


    return X_train, y_train, X_test, y_test,max_in,min_in,max_ra,min_ra,max_out,min_out


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
    return model


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


def solve_radiator(lr, intemp,max_in,min_in,max_ra,min_ra,max_out,min_out):
    """
     Figure out the optimum curve from outside temperature to battery network temperature
     in the way that inside temperature is a constant value
    """
    ls_pre = []
    ls_ratemp = []
    ls_outtemp = []
    for outtemp in range(-25,25,5):
        # for ratemp in range(round(min_ra*10),round(max_ra*10)):
        for ratemp in range(round(20 * 10), round(70 * 10)):
    # try some real data
    # for outtemp, ratemp in [[-20,630],[-10,530],[0,420],[10,330],[20,220]]:
            ratemp = ratemp/10

            rascl = (ratemp-min_ra)/(max_ra-min_ra)
            outscl = (outtemp - min_out) / (max_out - min_out)
            inscl = (intemp-min_in) / (max_in-min_in)
            # rascl = ratemp
            # outscl = outtemp
            # inscl = intemp


            pre = lr.predict( [[outscl,rascl,inscl]])
            ls_pre.append(pre[0])

            # # 2f
            if round(pre[0],2) == intemp:
                ls_ratemp.append(ratemp)
                ls_outtemp.append(outtemp)

    return ls_outtemp,ls_ratemp

def preprocess_rot( raw_data):
    '''
    Function for preprocessing data for modeling. Make outside temperature and radiator network temperature data unique
    before plotting.
    Inputs: Data frames
    Output: Training input data, training target data, testing input data, testing target data,

    '''
    raw_data = raw_data.copy()
    raw_data.rename(columns={'timestamp': 'Time', }, inplace=True)

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
    raw_data = raw_data.drop(todrop, axis=0, )
    todrop = raw_data[raw_data['Energy_consumption'].isna()].index.values

    raw_data = raw_data.drop(todrop, axis=0, )
    todrop = raw_data[raw_data['Outside_temperature_average'].isna()].index.values

    raw_data = raw_data.drop(todrop, axis=0, )
    raw_data = raw_data.reset_index(drop=True)



    df= raw_data.loc[:,['Outside_temperature_average','Radiator_network_1_temperature']]

    # unique outside temperature average
    unq = df['Outside_temperature_average'].unique()

    for i in unq:
        idx = df['Outside_temperature_average'][df['Outside_temperature_average'].values == i].index

        df.loc[idx[0],'Radiator_network_1_temperature'] =np.mean(df.loc[idx,'Radiator_network_1_temperature'])
        df.drop(idx[1:],inplace = True)


    df_train = df.loc[:int(len(raw_data) * 0.8),'Outside_temperature_average'].copy()
    df_test = df.loc[int(len(raw_data) * 0.8):,'Outside_temperature_average'].copy()




    df_train=pd.DataFrame(df_train,columns=['Outside_temperature_average'])
    df_test=pd.DataFrame(df_test,columns=['Outside_temperature_average'])


    X_train = df_train.loc[:,'Outside_temperature_average']
    y_train = df.loc[:int(len(raw_data) * 0.8),'Radiator_network_1_temperature']
    X_test = df_test.loc[:,'Outside_temperature_average']
    y_test = df.loc[int(len(raw_data) * 0.8):,'Radiator_network_1_temperature']


    return X_train, y_train, X_test, y_test,

def piecewise_linear(x, x0, y0, k1, ):
    """
    set the curve
    """
	# x<x0 ⇒ lambda x: k1*x + y0 - k1*x0
	# x>=x0 ⇒ lambda x: k2*x + y0 - k2*x0

    piece =  np.piecewise(x, [x<x0, x >= x0], [lambda x:k1*x + y0-k1*x0,lambda x:  y0])
    # piece =  np.piecewise(x, [x<x0, x >= x0], [lambda x:k1*x + y0-k1*x0,lambda x:  k1*x + y0-k1*x0])
    return piece

def sta(raw_data):
    """
    find relationship between radiator and outside temperature from statistics.
    """

    X_train, y_train, X_test, y_test, = preprocess_rot(raw_data)

    x = X_train.values.squeeze()
    y = y_train.values.squeeze()

    p, e = optimize.curve_fit(piecewise_linear, x, y)

    return x,y,piecewise_linear(x, *p)


def plot(raw_data,max_in,min_in,max_ra,min_ra,max_out,min_out,lr,intemp):
    """

    plot Relationship between outside temperature and radiaor temperature from both statistics and prediction
    """

    ls_outtemp, ls_ratemp = solve_radiator(lr,intemp,max_in,min_in,max_ra,min_ra,max_out,min_out)
    x, y, piece = sta(raw_data,)


    plt.figure()

    plt.title("Relationship between outside temperature and radiaor temperature")

    plt.plot(ls_outtemp, ls_ratemp,"r",label="predicted({}℃)".format(intemp))

    plt.plot(x, piece,"b",label = "measurement")
    plt.ylabel("Radiaor temperature")
    plt.xlabel('Outside temperature')
    plt.legend(loc="best")
    plt.grid()
    plt.show()



#main
# Download the downsampled data frame from csv-file.
raw_data = pd.read_csv(r'3452_building_data_0827.csv')
X_train, y_train, X_test, y_test,max_in,min_in,max_ra,min_ra,max_out,min_out = preprocess(raw_data)
lr= build_lr(X_train, y_train)

# meature inside temperature every f(int,>1) hours
f=[4,]


#set inside temperature
intemp = 22

# # plot
plot(raw_data,max_in,min_in,max_ra,min_ra,max_out,min_out,lr,intemp)
