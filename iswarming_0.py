import datetime
import pytz

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost
from sklearn import metrics
from statsmodels.formula.api import ols

import scipy.stats as stats

from readdata import *


time = []
oat = np.array([],dtype=float)

time, oat = load_data(time, oat, "data/2018555_1948.csv")
time, oat = load_data(time, oat, "data/2018554_1957.csv")

time, oat = load_data(time, oat, "data/2018553_1966.csv")
time, oat = load_data(time, oat, "data/2018551_1975.csv")
time, oat = load_data(time, oat, "data/2018550_1984.csv")
time, oat = load_data(time, oat, "data/2018547_1993.csv")
time, oat = load_data(time, oat, "data/2018546_2002.csv")
time, oat = load_data(time, oat, "data/2018544_2011.csv")
time, oat = load_data(time, oat, "data/2010360_2020.csv")




oat = (oat-32)*5/9
t = string_to_unixtime_array(time)
hod, doy = find_hod_doy_array(time)

#Remove nan values:
ids = np.argwhere(np.isnan(oat))
oat = np.delete(oat, ids)
t = np.delete(t, ids)
hod = np.delete(hod, ids)
doy = np.delete(doy, ids)

#Change the time scale:
#t = t/(60*60)   # to hours
#t = t/(60*60*24)    #to days
t = t/(60*60*24*365.25)   #to years 

t = t - t[0]



plt.plot(t, oat)
plt.title('Raw Data')
plt.xlabel('Time (year)')
plt.ylabel('Temperature (C)')
plt.show()



plt.plot(t[-48:], oat[-48:], label='temperature')
plt.plot(t[-48:], hod[-48:], label='hour of day')
plt.plot(t[-48:], doy[-48:], label='day of year')
plt.xlabel('Time (year)')
plt.title('Raw Data, Most Recent')
plt.legend()
plt.show()


#form X and y for xgboost
N = len(t)
X = np.hstack( ( hod.reshape(N,1), doy.reshape(N,1) ) )

regressor = xgboost.XGBRegressor(n_estimator=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)

regressor.fit(X, oat)

oat_hat = regressor.predict(X)

R2 = metrics.r2_score(oat, oat_hat)
MSE = metrics.mean_squared_error(oat, oat_hat)

res = oat - oat_hat

plt.plot(t, oat, label = 'Actual')
plt.plot(t, oat_hat, label = 'Predicted')
plt.title('Compare Actual OAT and Predicted OAT')
plt.xlabel('Time (year)')
plt.ylabel('Temperature (C)')
plt.legend()
plt.show()


plt.plot(t, oat_hat, label='Predicted')
plt.plot(t, res, label='Residue = Predicted - Actual')
plt.title('Raw Data')
plt.xlabel('Time (year)')
plt.ylabel('Residuel Temperature (C)')
plt.legend()
plt.show()



print('R2={}'.format(R2))
print('MSE = {}'.format(MSE))




plt.hist(oat, bins = 100)
plt.title('Actual Temperature')
plt.show()

plt.hist(res, bins = 100)
plt.title('Residual Temperature')
plt.show()

#This shows that res is normal distributed but actual temperature is not quite normal distributed

#do inear regression on residue and perform F test



data = pd.DataFrame({'x': t, 'y': res})
model = ols("y ~ x", data).fit()
print(model.summary())




