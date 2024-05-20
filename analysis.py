import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
import os
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)
#Read dataset
df_txn = pd.read_csv(f'{os.getcwd()}\\data\\transactions.csv')
df_oil = pd.read_csv(f'{os.getcwd()}\\data\\oil.csv')
df_str = pd.read_csv(f'{os.getcwd()}\\data\\stores.csv')
df_hol = pd.read_csv(f'{os.getcwd()}\\data\\holidays_events.csv')
df_train = pd.read_csv(f'{os.getcwd()}\\data\\train.csv')


#There are missing values in daily oil price, There are some approaches to tackel that.
#we imputed the missing values with average previous and next day and if there is first, would be next day
#if the last day would be a day before that
for index, row in df_oil.iterrows():
    if pd.isna(row['dcoilwtico']):
        if 0 < index < len(df_oil)-1:
            df_oil.at[index,'dcoilwtico']=df_oil['dcoilwtico'].iloc[index-1]
        elif index == 0:
            df_oil.at[index,'dcoilwtico']=df_oil['dcoilwtico'].iloc[index+1]
        elif index == len(df_oil)-1:
            df_oil.at[index,'dcoilwtico']=df_oil['dcoilwtico'].iloc[index-1]
df_oil.rename(columns={'dcoilwtico':'oil_price'},inplace=True)
df_train.set_index('date',inplace=True)
df_train.index.freq='D'
df_oil.set_index('date',inplace=True)
df_oil.index.freq='D'

#Join Train and oil to include daily oil price in training
df = df_train.join(df_oil)

#Join Holiday information and training information
df_hol_national = df_hol[df_hol['locale']=='National']
df_hol_national.drop_duplicates(subset='date',keep='last',inplace=True)
df_hol_national.reset_index(inplace=True,drop=True)

df_hol_local = df_hol[df_hol['locale']=='Local']
df_hol_local.drop_duplicates(subset=['date','locale_name'],keep='last',inplace=True)
df_hol_local.reset_index(inplace=True,drop=True)
df_hol_local.rename(columns={"locale_name":"city"},inplace=True)

df_hol_reg = df_hol[df_hol['locale']=='Regional']
df_hol_reg.drop_duplicates(subset=['date','locale_name'],keep='last',inplace=True)
df_hol_reg.rename(columns={"locale_name":"state"},inplace=True)

#Using update function alongside merge function
df_train_1 = df_train.copy()
df_train_1 = df_train_1.merge(df_hol_national,on='date',how='left')
df_train_2 = df_train.copy()
df_train_2 = df_train_2.merge(df_hol_local,on=['date','city'],how='left')
df_train_3 = df_train.copy()
df_train_3 = df_train_3.merge(df_hol_reg,on=['date','state'],how='left')
df_train_1.set_index(['id', 'date'], inplace=True)
df_train_2.set_index(['id', 'date'], inplace=True)
df_train_3.set_index(['id', 'date'], inplace=True)
df_train_1.update(df_train_2)
df_train_1.update(df_train_3)
df_train_1.reset_index(inplace=True)

#df_train_1['date']=df_train_1['date'].apply(lambda x: pd.to_datetime(x))
#faster approach
df_train_1['date'] = pd.to_datetime(df_train_1['date'])
df_train_1['day_of_week'] = df_train_1['date'].dt.day_name()
# df_train_1['year']=df_train_1['date'].apply(lambda x: x.year)
# df_train_1['month']=df_train_1['date'].apply(lambda x: x.month)
#Faster
df_train_1['year'] = df_train_1['date'].dt.year
df_train_1['month'] = df_train_1['date'].dt.month


#resampling method is used to do Time resampling for example get mean of
#sales in weekly, monthly, quartly, annually. 
#Note: the index must be datetime
#Quartly
#df_train_1['sales'].resample(rule='Q').mean()
#Annualy
#df_train_1['sales'].resample(rule='A').mean()

#Time shifting can be used to shift rows based on specific numbers or
#if our index is time we can use it to shift 
#df_train_1.shift(periods=1,freq='M').head()
#For example, this will shift to end of each month

#By the following code we can rolling data based on specific window
#and get the mean, max, min or sum of it
# df_train_1['sales'].rolling(window=90).mean().plot()
#the above code is considered 90 days window and calculate the mean of 
#sales for each 90 days. It is usefull to show and 
# df_train_1['Close:30day mean'] = df_train_1['sales'].rolling(window=30).mean()

#Expanding is used to to get aggragate2 function of specific threshold
#for example, by expanding of 2 days, it get the mean of first 2 days,
#then first 4 days, then first 6days, then first 8 days and so on and so forth.
# ax = df_train_1['sales'].plot(xlim=['2013-01-01','2013-03-01'],figsize=(20,8))
# ax.set(xlabel='')
# ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=0))
# ax.xaxis.set_major_formatter(dates.DateFormatter('%a-%B-%d'))
#by the above functions, it allows us to change x labels to just show
#starting day of week which can be monday (0)
#Look at date formatting method documentation to do more examples

#Total Monthly sales
monthly_sales = pd.DataFrame(df_train_1['sales'].resample(rule='M').sum())

sales = monthly_sales
#Doing some time series analysis by statsmodel
from statsmodels.tsa.filters.hp_filter import hpfilter
#First considering holter prescott filtering which devided the time series (y_t)
#To trend (T_t) and cyclical (C_t) components by using smoothing factor (lambda)
#lambda for monthly data is 129,600 for quartly is 1600 and annually is 6.25
#the results of hpfilter is tuple which first is cycle and second is trend
# sales_cycle, sales_trend = hpfilter(monthly_sales['sales'],lamb=129_600)
# monthly_sales['sales_trend'] = sales_trend
# monthly_sales.plot(figsize=(12,8))

#To do ETS Decomposition in order to check if there is seasonality, trend, error
#it is possible to consider stats model and seasonality of that
#First step, we aggragate sales data into the monthly bases in the aboved section
# from statsmodels.tsa.seasonal import seasonal_decompose
# result = seasonal_decompose(monthly_sales['sales'],model='multiplicative')
#To change size of plot can be used pylab modules as follows
# from pylab import rcParams
# rcParams['figure.figsize']=12,8
# result.plot()

#The aim was to convert raw time series data into its decompisiton of
#Trend, Seasonality and REsidual or Error. 


#The simple moving average approach:
# monthly_sales['6_month_SMA'] = monthly_sales['sales'].rolling(window=6).mean()
# monthly_sales['12_month_SMA'] = monthly_sales['sales'].rolling(window=12).mean()
# monthly_sales.plot(figsize=(12,8))

#The Explonationaly weighted moving average (EWMA) 
#There are span, center of mass, halflife method with specific value
#for example easier one is span and following consider 12 months period
# monthly_sales['EWMA-12']=monthly_sales['sales'].ewm().mean()
# monthly_sales[['sales','EWMA-12']].plot(figsize=(12,8))

# Span = 12
# alpha = 2/(Span+1)

# from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
# span =12
# alpha = 2/(span+1)
# monthly_sales['SES-12']=SimpleExpSmoothing(monthly_sales['sales']).fit(smoothing_level=alpha, optimized=False).fittedvalues.shift(-1)
# monthly_sales['DES-mul-12']=ExponentialSmoothing(monthly_sales['sales'],trend='mul').fit().fittedvalues.shift(-1)
# monthly_sales['DES-add-12']=ExponentialSmoothing(monthly_sales['sales'],trend='add').fit().fittedvalues.shift(-1)
# monthly_sales['TES-mul-12']=ExponentialSmoothing(monthly_sales['sales'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues

# monthly_sales[['sales','SES-12', 'DES-mul-12', 'DES-add-12', 'TES-mul-12']].plot(figsize=(20,10))


'''
Forcasting Time Series
'''
from statsmodels.tsa.stattools import adfuller

def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")