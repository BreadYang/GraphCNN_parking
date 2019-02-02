#################
# Multilayer LSTM for parking prediction
# Using Keras
#################


import matplotlib as mpl
mpl.use('Agg')
from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv 
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
import numpy as np
from sklearn.metrics import mean_squared_error
from numpy import linalg as LA

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(n_out, 0, -1):
        cols.append(df.shift(-i))
    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def get_average(data, ave_num):
    size, dim = data.shape
    size_after = size/ave_num
    df = []
    idx = 0
    tmprow = []
    for row in data:
        if idx == 0:
            tmprow = row
            idx += 1
        elif idx < ave_num:
            tmprow += row
            idx += 1
        if idx == ave_num:
            idx = 0
            df.append((tmprow.astype('float32'))/ave_num)
            tmprow = []
    return np.array(df)


def augmentation(data, y, folds, sigma):
    output = data.copy()
    y_noise = y.copy()
    for idx in range(folds):
        tmp = np.random.normal(0, sigma, data.shape) + data
        output = np.append(tmp, output, axis=0)
        output = data.copy()
    for idx in range(folds):
        tmp = np.random.normal(0, sigma, y.shape) + y
        y_noise = np.append(tmp, y_noise, axis=0)
    return output, y_noise


# In[159]:
def forecast_lstm(model, X, forcast_steps):
    output = []
    Xtmp = X.copy()
    for i in range(forcast_steps):
        forecast = model.predict(np.expand_dims(Xtmp.copy(), axis=0))
        output.append(forecast[0])
        Xtmp = np.append(Xtmp[1:,], forecast, axis=0)
    # convert to array
    return np.array(output)


# In[43]:


weather = read_csv('weather_feature.csv', header=None)
spd = read_csv('spd_features.csv', header=None)
parking = read_csv('parking_feature.csv', header=None)
Y = read_csv('Y.csv', header=None)
weather = weather.drop(weather.columns[0], axis=1).values
spd = spd.drop(spd.columns[0], axis=1).values
parking = parking.drop(parking.columns[0], axis=1).values
Y = (Y.drop(Y.columns[0], axis=1)).values

weather_dim = weather.shape[1]
# pd.merge(dfsingle, spd, left_on=['Timestamp'], right_on=['Timestamp'])
# pd.merge(dfsingle, spd, left_on=['Timestamp'], right_on=['Timestamp'])


rounding = 1
padding = 5
parking = parking.reshape(parking.shape[0], padding, 38)
spd = spd.reshape(spd.shape[0], padding, 38)
#weather = np.repeat(weather, padding, axis=1)
#weather = weather.reshape(weather.shape[0], padding, weather_dim)
Data = (np.concatenate((parking, spd), axis=2))
n_features = spd.shape[2]+parking.shape[2]

print(Data.shape)

scaler = MinMaxScaler(feature_range=(-1, 1))
X_norm = scaler.fit_transform(Data.reshape(Data.shape[0], padding*Data.shape[2]))

scaler1 = MinMaxScaler(feature_range=(-1, 1))
y_norm = scaler1.fit_transform(Y)
# specify the number of lag hours


# In[36]:


def run_model(hidden_1 = 256, hidden_2 = 32, hidden_3 = 128, pref = ''):

    checkpointfile = pref+'model_save.hdf5'
    logfile = pref+'training_log.csv'
    batch_size = 100
    noise_sigma = 0.01
    data_gen = 50
    train_X, test_X, train_y, test_y = train_test_split(X_norm, y_norm,
            test_size=.2, random_state=332)
    train, test = augmentation(train_X, train_y, data_gen, noise_sigma)
    train_X = train_X.reshape((train_X.shape[0], padding, n_features))
    test_X = test_X.reshape((test_X.shape[0], padding, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    model = Sequential()
    model.add(LSTM(hidden_1, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
    if hidden_3 != 0:
        model.add(LSTM(hidden_2, return_sequences=True,))  # return a single vector of dimension 32
        model.add(LSTM(hidden_3))
    else:
        model.add(LSTM(hidden_2))
    model.add(Dropout(0.5))  
    model.add(Dense(38))
    sgd = optimizers.SGD(lr=0.01, decay=1e-2, momentum=0.5, nesterov=True)
    model.compile(loss='mse', optimizer=sgd)

    # fit network
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, verbose=0, mode='auto')
    checkpoint = ModelCheckpoint(checkpointfile, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=2)

    history = model.fit(train_X, train_y, epochs=500, batch_size=batch_size, validation_data=(test_X, test_y),
    verbose=2, shuffle=True, callbacks=[early_stopping, checkpoint])

    val_loss = history.history['val_loss']
    train_loss = history.history['loss']
    yhat_deno = scaler1.inverse_transform(model.predict(test_X))
    testy_deno = scaler1.inverse_transform(test_y)
    mse = mean_squared_error(yhat_deno, testy_deno)
    print(mse)

    plt.figure()
    plt.plot(train_loss, label='Training')
    plt.plot(val_loss, label='Validation')
    plt.legend()
    plt.ylabel('MSE')
    plt.xlabel('epoch')
    plt.title('Training progress, final mse'+ str(mse))
    title = 'Parking_'+'_'.join(str(i) for i in [padding, hidden_1, hidden_2, hidden_3])+'.png'
    plt.savefig(title)

    with open(logfile, 'a') as f:
        outputls = ['epochs:', len(history.history['loss']), "hidden_1:", hidden_1, "hidden_2:", hidden_2, "hidden_3:", hidden_3, "MSE:", mse]
        output = ','.join(str(i) for i in outputls)+'\n'
        f.write(output)


# In[ ]:


pref = 'parking'
# run_model(12, 0, 256, 64)
# run_model(24, 0, 512, 64)
hidden_1 = 512
### main
for hidden_2 in [512, 1024]:
    for hidden_1 in [1024, 2048]:
        for hidden_3 in [0, 512, 128]:
            run_model(hidden_1, hidden_2, hidden_3, pref)

