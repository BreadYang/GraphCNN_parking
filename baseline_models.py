############
# LASSO and other baseline models for parking prediction
############

import csv
from collections import OrderedDict
from math import sin, cos, sqrt, atan2
from datetime import datetime, timedelta
from scipy import stats
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from operator import add

hours_pred = [10, 11, 12, 13, 14, 15, 16, 17]
train_ratio = 0.75
def random_train_test_dates(Data, Label, test_size=.2, random_state=332):
    dates = {}
    for day in Label['Time'].dt.date:
        if day not in dates:
            dates[day] = 1
    dates_ary = dates.keys()
    train, test = train_test_split(dates_ary, test_size=test_size, random_state=random_state)

    train_data = Data.loc[(Data['Time'].dt.date.isin(train)), Data.columns].drop(['Time'], axis=1)
    test_data = Data.loc[(Data['Time'].dt.date.isin(test)), Data.columns].drop(['Time'], axis=1)

    train_Label = Label.loc[(Data['Time'].dt.date.isin(train)), Label.columns].drop(['Time'], axis=1)
    test_Label = Label.loc[(Data['Time'].dt.date.isin(test)), Label.columns].drop(['Time'], axis=1)
    return train_data, test_data, train_Label, test_Label

weather = read_csv('weather_feature.csv')
spd = read_csv('spd_features.csv')
parking = read_csv('parking_feature.csv')
Y = read_csv('Y_win.csv')

weather['Time'] = pd.to_datetime(weather['Time'], format= "%m/%d/%y %H:%M")
spd['Time'] = pd.to_datetime(spd['Time'], format= "%m/%d/%y %H:%M")
parking['Time'] = pd.to_datetime(parking['Time'], format= "%m/%d/%y %H:%M")
weather['Time'] = pd.to_datetime(weather['Time'], format= "%m/%d/%y %H:%M")
Y['Time'] = pd.to_datetime(Y['Time'], format= "%m/%d/%y %H:%M")
spd = spd.drop(['Time'], axis=1)


Data = (pd.concat([parking, spd], axis=1))
# Data.columns = [str(x) for x in range(Data.shape[1])]

##### LASSO
errors = []
errors_train = []
pktotal = [0,0,0,0,0]
spdtotal = [0,0,0,0,0]
linktotal = np.zeros(38)
for Link_id in Y.columns[1:]:
    y = Y[['Time', str(Link_id)]]
    train_X, test_X, train_y, test_y = random_train_test_dates(Data, y, test_size=.2, random_state=332)
    model=linear_model.Lasso(alpha=0.1).fit(train_X, train_y)
    summary = dict(zip(Data.columns, model.coef_))
    for key in summary.keys():
        if summary[key] == 0.0:
            del summary[key]
            
    ##### Sum keys
    pkcnt = [0,0,0,0,0]
    spdcnt = [0,0,0,0,0]
    linkcnt = np.zeros(38)
    for key in summary.keys():
        dataset = 'Park' if '04' in key else 'Speed'
        if '04' in key:
            pkcnt[0]  += 1
        else:
            spdcnt[8-5] +=1
        pktotal = map(add, pktotal, pkcnt)
        spdtotal = map(add, spdtotal, spdcnt)
        linktotal = map(add, linktotal, linkcnt)
#     print "Link idx:", Link_id
#     print "Parking feature cnt per T:", pkcnt
#     print "Spd feature cnt per T:", spdcnt
#     print "Link Feture cnt per idx", linkcnt
#     ######
    error = mean_squared_error(model.predict(test_X), test_y)
    errors.append(error)
    errors_train.append(mean_squared_error(model.predict(train_X), train_y))
#     break
#     print model.score(train_X, train_y), model.score(test_X, test_y)
print errors
print ("LASSO mse:", np.mean(errors_train), np.mean(errors))

Data = (pd.concat([parking, spd], axis=1))
Data = Data.drop(['Time'], axis=1)
Y = Y.drop(['Time'], axis=1)
Data.columns = [str(x) for x in range(Data.shape[1])]
Y.columns = [str(x) for x in range(Y.shape[1])]

### HA
errors = []
error_train = []
for Link_id in range(Y.shape[1]):
    y = Y[str(Link_id)]
    train_X, test_X, train_y, test_y = train_test_split(Data, y,
        test_size=.2, random_state=332)
    np.array(test_X)[:, Link_id]
    pre = np.average(np.array(test_X)[:, Link_id])
    error = mean_squared_error(np.repeat(pre, test_y.shape[0]), test_y)
    errors.append(error)
    error_train.append(mean_squared_error(np.repeat(pre, train_y.shape[0]), train_y))
#     print model.score(train_X, train_y), model.score(test_X, test_y)
print ("HA mse: ", np.mean(error_train), np.mean(errors))

### Last
errors = []
error_train = []
for Link_id in range(Y.shape[1]):
    y = Y[str(Link_id)]
    train_X, test_X, train_y, test_y = train_test_split(Data, y,
        test_size=.2, random_state=332)
    np.array(test_X)[:, Link_id]
    error = mean_squared_error(np.array(test_X)[:, Link_id], test_y)
    errors.append(error)
    error_train.append(mean_squared_error(np.array(train_X)[:, Link_id], train_y))
#     print model.score(train_X, train_y), model.score(test_X, test_y)
print ("Last mse: ", np.mean(error_train), np.mean(errors))