#!/usr/bin/env python
# coding: utf-8

# This time series modeling analysis using Python will be used to identify the trends, seasonality, and irregularity in revenue over time. 
# 
# My research question for this analysis is “Can I accurately predict revenue trends over time with a time series analysis?” 

# The main objective and goal of this research analysis are to give stakeholders in a telecommunications company data that can be used to determine the seasonality of data in order to better determine times where they are more profitable historically. In doing so they can predict trends in profitability so that they can better influence profits in the future and identify possible customer churn by looking at drops in revenue.

# One assumption of time series analysis is that time series analysis data is stationary whereas the mean, and variance do not change over time. In addition, an assumption exists that there is constant autocorrelation meaning the relationship to the variables in the time series model is constant. 

# In[1]:


#imports for time series analysis 
import pandas as pd
from pandas import to_datetime

import os 

import numpy as np

import datetime
from datetime import datetime 

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm

from scipy import signal

from pylab import rcParams

plt.style.use('ggplot')

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse

from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

import pmdarima as pm
from pmdarima import model_selection 
from pmdarima.model_selection import train_test_split 
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

import warnings
warnings.filterwarnings('ignore')


# In[2]:


#read csv into pandas dataframe
df = pd.read_csv('teleco_time_series.csv')


# Once the libraries and packages for the time series analysis are imported and the csv is read into the pandas database, the data is evaluated for null/NA and duplicate values, the shape of the data is obtained (number of rows, columns) and summary statistics are obtained. 

# In[3]:


#view top of the data
df.head()


# In[4]:


#dataset size
df.shape


# In[5]:


#view index and # of records present
print("The last index in the dataframe is", df.index[-1], "and there are", len(df), "records present.")
print("There are no gaps in the time sequence.")


# In[6]:


#data types
df.info()


# In[7]:


#null values
df_nulls = df.isnull().sum()
print(df_nulls)


# In[8]:


#duplicates
print(df.Day.duplicated().sum())


# In[9]:


#NAs
print(df.isna().sum())


# In[10]:


#summary stats
df.describe()


# The column “Day” which holds integer data types is the time step format of the dataset and there are no missing or duplicate values. 

# In[11]:


#save a clean dataset as csv-(done to save any changes made for duplicates or missing values)
df.to_csv('teleco_time_series_clean.csv')


# One of the first visual looks at the data is a time series line graph shown below which demonstrates an increasing trend and non-stationary data. 

# In[12]:


#plot the time series
df.plot()
plt.title("Line Graph of Revenue Over Time")
plt.ylabel("Revenue (in million $)")
plt.show();


# In[13]:


#line graph visualizing the realization of the time series
plt.figure(figsize=(18, 6))
plt.plot( df['Day'], df['Revenue'], color='tab:blue')
plt. xlabel('Time(Day)')
plt.ylabel('Revenue in millions, USD')
plt.title(' Revenue for The last two years')
plt.show()


# The data was evaluated for stationarity utilizing the augmented Dickey-Fuller test. This is a test that uses the null hypothesis that the values are nonstationary. The P-Value result in this test is what is used to determine whether to reject the null hypothesis or not. Here the P-value is more than 0.05 which tells us the mean and autocovariance are not stationary therefore the null hypothesis is not rejected. 

# In[14]:


#Augmented Dickey-Fuller unit root test
adft = adfuller(df.iloc[:, 1].values, autolag='AIC')
print("1. ADF c-value : ",adft[0])
print("2. P-Value : ", adft[1])
print("3. Num Of Lags : ", adft[2])
print("4. Num Of Observations Used :", adft[3])
print("5. Critical Values :")
for key, val in adft[4].items():
    print("\t",key, ": ", val)


# The data was also analyzed using rolling statistics. 

# In[15]:


#set up Stationairty test
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=30).mean()
    rolstd = timeseries.rolling(window=30).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)


# In[16]:


#perform test on entire dataset( again, the mean is not zero, hence dataset not stationary)
test_stationarity(df['Revenue'])


# In[17]:


#run stationarity test 
result = adfuller(df['Revenue'])

#print test statistic
print("The t-statistic is:", round(result[0], 2))

#print p-value
print("The p-value is:", round(result[1], 2))

#print critical values 
crit_vals = result[4]
print("The critical value of the t-statistic for a 95% confidence level is:", round(crit_vals['5%'], 2))


# The results of the analysis tell us our data is not stationary and will therefore require a transformation. The analysis specifically shows a t-statistic of -1.92 and a p-value of 0.32. To achieve a confidence level of 95%, we should reject the null hypothesis given the t-statistic should be below -2.87 and the p-value should be below 0.05. 

# The dataset is split. The split consists of 80% training and 20% testing data.  

# In[18]:


#split dataset into training and test sets with 80% for training(80% of 731 days= 585)
df['Day'] = df.index
df_train = df[:585]


# In[19]:


#20% Test (from row 585 to the end of the data)
df_test = df[585:]


# In[20]:


#plot training and test datasets 
plt.figure(figsize=(10, 4))
plt.plot(df_train['Revenue'], color='black')
plt.plot(df_test['Revenue'], color='blue')
plt.title('Train_Test Split ')
plt.xlabel('Time(Days)')
plt.ylabel('Revenue in Millions, USD')
plt.show()


# In[21]:


#decompose the training dataset
decomposition=seasonal_decompose(df_train['Revenue'], model='additive', period=30)
trend=decomposition.trend
seasonal=decomposition.seasonal
residual=decomposition.resid
plt.subplot(411)
plt.plot(df_train['Revenue'],color='red', label='original_series')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,color='blue', label='Trend')
plt.legend(loc='best')
plt.tight_layout()
plt.subplot(414)
plt.plot(residual,color='green', label='Residual')
plt.legend(loc='best')
plt.tight_layout()
plt.subplot(413)
plt.plot(seasonal,color='brown', label='Seasonality')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# In[22]:


#plot the trend component only
plt.figure(figsize=(12, 4))
plt.plot(trend, color='tab:blue')
plt.xlabel('Day')
plt.ylabel('Revenue Trend')
plt.title('Decomposed Revenue Trend')
plt.show()


# In[23]:


#plot autocorrelation function on training set
plot_acf(df_train['Revenue']);


# In[24]:


#plot partial autocorrelation function on training set
plot_pacf(df_train['Revenue'], method='ywm')


# In[25]:


#run spectral density function
plt.psd(df['Revenue'])


# In[26]:


#transform the original dataset by differencing to attain stationarity and remove trend
diff_df = np.diff(df['Revenue'], axis=0)


# In[27]:


#decompose and plot again after differencing
decomposition=seasonal_decompose(diff_df, model='additive', period=30)
trend=decomposition.trend
seasonal=decomposition.seasonal
residual=decomposition.resid
plt.subplot(411)
plt.plot(df_train['Revenue'],color='red', label='original_series')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,color='blue', label='Trend')
plt.legend(loc='best')
plt.tight_layout()
plt.subplot(414)
plt.plot(residual,color='green', label='Residual')
plt.legend(loc='best')
plt.tight_layout()
plt.subplot(413)
plt.plot(seasonal,color='brown', label='Seasonality')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# In[28]:


# Plot the trend component only again to confirm trend is eliminated
plt.figure(figsize=(12, 4))
plt.plot(trend, color='tab:blue')
plt.xlabel('Day')
plt.ylabel('Revenue Trend')
plt.title('Decomposed  and differenced Revenue Trend')
plt.show()


# In[29]:


#Plot the residuals( errors, basically zeros)
plt.plot(diff_df)
plt.show()


# In[30]:


# Run adfuller test on the differenced DataFrame again to check for stationarity.
#This time P-value is zero<0.05, hence data is stationary and ready for ARIMA

adft_diff = adfuller(diff_df, autolag='AIC')
print("1. ADF c-value : ",adft_diff[0])
print("2. P-Value : ", adft_diff[1])
print("3. Num Of Lags : ", adft_diff[2])
print("4. Num Of Observations Used :", adft_diff[3])
print("5. Critical Values :")
for key, val in adft[4].items():
    print("\t",key, ": ", val)


# In[31]:


#Test for stationarity again after differencing( confirmation, mean is virtually zero and constant)
diff_dff=pd.DataFrame(diff_df)
test_stationarity(diff_dff)


# In[32]:


# running the auto ARIMA model to find the best model that minimizes AIC.
arima_model = auto_arima(df['Revenue'], start_P=0,
                        start_q=0,
                        max_p=2,
                        max_q=2,
                        m=30,
                        seasonal=True,
                        d=1,
                        D=1,
                        trace=True,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True)
arima_model.summary()


# In[33]:


# Build SARIMAX model on the train Data set using the (p,d,q)(P,D,Q)m results from the model above
model_SAR = sm.tsa.SARIMAX(df_train['Revenue'], order=(1, 1, 0), seasonal_order=(2, 1, 0, 30))
SARIMAX_Results = model_SAR.fit()

# Print results tables
print(SARIMAX_Results.summary())


# In[34]:


# call out forecast function
result_SAR = SARIMAX_Results.get_forecast()


# In[35]:


# prediction on test set( 20% test, 80% train)

predictions = SARIMAX_Results.predict(585, 730, typ = 'levels').rename('Predictions')

#Predict with respect to test set
plt.figure(figsize=(14, 6))
plt.plot(df_train['Revenue'], 'o-', color='black', label = '80% Train Dataset')
plt.plot(df_test['Revenue'], 'o-',  color='blue', label = '20% Test Dataset')
plt.plot(predictions, 'o-', color='red', label = 'Predictions')
plt.title('Teleco Revenue Predictions')
plt.xlabel('Day')
plt.ylabel('Revenue, USD')
plt.legend(loc='best', fontsize = 8)
plt.show()


# In[36]:


#Summary computations on test set
test_first= df_test['Revenue'].values.astype('float32')
forecast_test = result_SAR.predicted_mean


# In[37]:


print('Expected : %.2f' % forecast_test)
print('Forecasted : %.2f' % test_first[0])
print('Standard Error : %.2f' % result_SAR.se_mean)


# In[38]:


# confident intervals
intervals = [0.2, 0.1, 0.05, 0.01]
for a in intervals:
                ci = result_SAR.conf_int(alpha=a)
                print('%.1f%% Confidence Level: %.2f between %.2f and %.2f' % ((1 - a) * 100, forecast_test, ci['lower Revenue'], ci['upper Revenue']))
ci


# In[39]:


# Run Mean Squared Error
MSE = mean_squared_error(df_test['Revenue'], predictions)
print('Summary')
print('MSE: ', round(MSE, 4))

# Run Root Mean Squared Error
RMSE = rmse(df_test['Revenue'], predictions)
print('RMSE: ', round(RMSE, 4))


# In[40]:


# make predictions with respect to the complete dataset
model00 = sm.tsa.statespace.SARIMAX(df['Revenue'],order=(1, 1, 0), seasonal_order=(2, 1, 0, 30))
results00 = model00.fit()
# Print results tables
print(results00.summary())


# In[41]:


# Forecast for the following quarter on the entire dataset( 2 years=731, 822=731+ 91, 91=quarter of 3 months)
forecast00 = results00.predict(731, 822, typ = 'level').rename('Teleco Forecast')


# plot predicted values
plt.figure(figsize=(14,6))
plt.plot(df['Revenue'], 'o-', color='black', label='Revenue Past')
plt.plot(forecast00, 'o-', color='red', label='Revenue Forecast' )
plt.legend(loc='best')
plt.show()


# In[42]:


# Plot all for comparison
plt.figure(figsize=(14, 6))
plt.plot(df_train['Revenue'], 'o-', color='black', label = '80% Train Dataset')
plt.plot(df_test['Revenue'], 'o-',  color='blue', label = '20% Test Dataset')
plt.plot(predictions, 'o-', color='green', label = 'Predictions on test set')
plt.plot(forecast00, 'o-', color='red', label='Revenue Forecast' )
plt.title('Teleco Revenue Predictions/forecast')
plt.xlabel('Day')
plt.ylabel('Revenue, USD')
plt.legend(loc='best', fontsize = 8)
plt.show()


# In[43]:


#Try ARIMA model instead of Seasonal ARIMA
model_arima = ARIMA(df['Revenue'], order=(1,1,0))  
results_ARIMA = model_arima.fit()
plt.plot(df['Revenue'], color='blue')
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.show()


# summary of fit model
print('summary of fit model', results_ARIMA.summary())


# line plot of residuals
residuals = pd.DataFrame(results_ARIMA.resid)
residuals.plot()
plt.title('line plot of residuals')
plt.show()


# density plot of residuals
residuals.plot(kind='kde')
plt.title('density plot of residuals')
plt.show()

# summary stats of residuals
print( 'summary stats of residuals', residuals.describe())


# ARIMA Selection
# For this analysis the selection of an ARIMA model was made after fitting the dataset. The best model was the model with results (1,1,0)(2,1,0)[30]. 

# Forecast Prediction Intervals 
# Four confidence intervals were tested in this analysis and are as follows: 
# 80.0% Confidence Level: 13.68 between 13.00 and 14.36
# 90.0% Confidence Level: 13.68 between 12.81 and 14.55
# 95.0% Confidence Level: 13.68 between 12.64 and 14.72
# 99.0% Confidence Level: 13.68 between 12.31 and 15.04
# 
# Evaluating the data with a 95% confidence level tells us there is 5% probability that the real observation will fall out of the range of 12.64 and 14.72. 

# Justification of the Forecast Length 
# The data was analyzed for one year of revenue predictions. This length was used in order to identify whether or not first the data would match the trend of increased profits in the previous years. Additional analysis could be performed to allow stakeholders the opportunity to gather additional forecasted predictions.

# The SARIMEX model was evaluated with mean square error (MSE) and root mean square error (RMSE). 
# Summary
# MSE:  4.5606
# RMSE:  2.1356
# 
# The results of the metric verification were small enough to conclude an accurate model. 

# Recommendations
# 
# The above analysis suggests the ARIMA model is a good test and that data in revenue shows upward trends over time.(Prabhakaran, 2022)The forecast for the next two years shows an expected increase in revenue. 
# In addition, the SARIMA model used to determine the seasonality of the data shows a 0.36% probability indicating there is not much seasonality identified in this model for this time series analysis. It is not recommended to use this dataset for seasonality information on business needs.

# References
# 
# Prabhakaran, S. (2022, April 4). Time series analysis in python - A comprehensive guide with examples - ML+. Machine Learning Plus. Retrieved May 28, 2022, from https://www.machinelearningplus.com/time-series/time-series-analysis-python/ 
# 
# Simplilearn. (2021, September 15). Understanding time series analysis in Python. Simplilearn.com. Retrieved May 25, 2022, from https://www.simplilearn.com/tutorials/python-tutorial/time-series-analysis-in-python 

# In[ ]:




