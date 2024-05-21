import pandas as pd
import numpy as np
import os
from utils import download_competition_data, load_kaggle_data, data_loading, TS_chart, Holiday
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg, AR
from statsmodels.tools.eval_measures import rmse
from pmdarima import auto_arima
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults, ARIMAResultsWrapper
from statsmodels.tsa.api import SARIMAX


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
        return True
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")
        return False
    
def AR_Model(df_TS: pd.DataFrame, nobs: int, lag:int):
    '''
    Function will train AR models with different requested lags and predict the time series variable.
    Input df_TS: time series data to model
    Input nobs: number of observation as test
    lags: list of number of previous data points would be consider to predict a future value
    '''
    train = df_TS.iloc[:-nobs]
    test = df_TS.iloc[-nobs:]
    AR = AutoReg(train['sales'],lags=lag).fit()
    prediction = AR.predict(start=len(train),end=len(train)+len(test)-1)
    AR_performance = {'RMSE':rmse(test['sales'],prediction),f"AR{lag}_parameter":lag}
    print(prediction.to_list())
    return AR_performance, prediction.to_list()

def ARIMA_Model(df_TS: pd.DataFrame, nobs: int,exog_features:list=[]):
    '''
    Function will train ARIMA models with respective ARIMA order parameter (p,d,q)
    Input df_TS: time series data to model
    Input nobs: number of observation and to predict
    '''
    if len(exog_features) == 0:
        
        train = df_TS.iloc[:-nobs]
        test = df_TS.iloc[-nobs:]
        arima_parameters = auto_arima(df_TS['TS'],seasonal=False,maxiter=1000)
        arima_order = arima_parameters.get_params()['order']
        model = ARIMA(train['sales'],order=arima_order,enforce_invertibility=False)
        result = model.fit()
        prediction = result.predict(start=len(train),end=len(train)+len(test)-1)
        ARIMA_performance = {'RMSE':rmse(test['sales'],prediction),"arima_order":arima_order}
        print(prediction.to_list())
    else:
        
        train = df_TS.iloc[:-nobs]
        test = df_TS.iloc[-nobs:]
        arima_parameters = auto_arima(df_TS['TS'],seasonal=False,maxiter=1000)
        arima_order = arima_parameters.get_params()['order']
        model = ARIMA(train['sales'],exog=train[exog_features],order=arima_order,enforce_invertibility=False)
        result = model.fit()
        prediction = result.predict(start=len(train),end=len(train)+len(test)-1,exog=test[exog_features])
        ARIMA_performance = {'RMSE':rmse(test['sales'],prediction),"arimax_order":arima_order}
        print(prediction.to_list())
    return ARIMA_performance, prediction.to_list()


                             
def SARIMA_Model(df_TS: pd.DataFrame, nobs:int,exog_features:list=[]):
    '''
    Function will train SARIMA models with respective ARIMA order parameter (p,d,q) and Seasonal parameter (P,D,Q)
    Input df_TS: time series data to model
    Input nobs: number of observation and to predict
    '''
    if len(exog_features) == 0:
        
        train = df_TS.iloc[:-nobs]
        test = df_TS.iloc[-nobs:]
        sarima_parameters = auto_arima(df_TS['TS'],seasonal=False,maxiter=1000)
        arima_order = sarima_parameters.get_params()['order']
        seasonal_order = sarima_parameters.get_params()['seasonal_order']
        model = SARIMAX(train['sales'],order=arima_order,seasonal_order=seasonal_order,
                        enforce_invertibility=False)
        result = model.fit()
        prediction = result.predict(start=len(train),end=len(train)+len(test)-1)
        SARIMA_performance = {'RMSE':rmse(test['sales'],pred2),
                              "order":arima_order,
                              "seasonal_order":seasonal_order}
        print(prediction.to_list())
    else:
        
        train = df_TS.iloc[:-nobs]
        test = df_TS.iloc[-nobs:]
        sarima_parameters = auto_arima(df_TS['TS'],seasonal=False,maxiter=1000)
        arima_order = sarima_parameters.get_params()['order']
        seasonal_order = sarima_parameters.get_params()['seasonal_order']
        model = SARIMAX(train['sales'],exog=train[exog_features],
                        order=arima_order,seasonal_order=seasonal_order,enforce_invertibility=False)
        result = model.fit()
        prediction = result.predict(start=len(train),end=len(train)+len(test)-1)
        SARIMA_performance = {'RMSE':rmse(test['sales'],pred2),
                              "order":arima_order,
                              "seasonal_order":seasonal_order,
                              "Exog_features":['Holiday','onpromotion']}
        print(prediction.to_list())
    return SARIMA_performance, prediction.to_list() 

def select_model(df: pd.DataFrame,df_oil: pd.DataFrame, nobs: int,family:list,store:list):
    '''
    This function runs various Time series models such as AutoReg, ARIMA, ARIMAX, SARIMAX, VARMAX for each combination
    of store and family items.
    Input df: Sales time series
    Input nobs: number of obervation in the future which will be used as test
    '''
    if (len(family) == 0) and (len(store) ==0):
        family_items = list(df['family'].unique())
        store_number = list(df['store_nbr'].unique())
    else:
        family_items = family
        store_number = store
    
    predict_accuracy = {}
    
    for item in family_items:
        for number in store_number:
            df_temp = df[(df['family'].str.lower()==item.lower()) & (df['store_nbr']==number)][['id','date','sales','onpromotion','Holiday']]
            df_temp = df_temp.merge(df_oil,on='date',how='left')
            df_temp['oil_price'].fillna(method='ffill', inplace=True)
            df_temp['Holiday'] = df_temp['Holiday'].apply(lambda x: 1 if x=='True' else 0)
            df_temp.set_index('date',inplace=True)
            
            #Do stationary test: if time series is stationary, consider simpler models such as AR
            #if the model is non stationary, consider more complex model such as ARIMA and SARIMA
            stationary_test = adf_test(df_temp['sales'])
            test = df_temp.iloc[-nobs:]
            model_name = ['sales']
            if stationary_test:
                
                for lag_value in [5,15,30,45,60,90]:
                    predict_accuracy[f'{item}_Store{number}_AR{lag_value}'], test[f"{item}_store{number}_AR{lag_value}"]=AR_Model(df_temp,nobs=nobs,lag=lag_value)
                    model_name.append(f"{item}_store{number}_AR{lag_value}")
            else:
                
                predict_accuracy[f"{item}_Store{number}_ARIMA"], test[f"{item}_store{number}_ARIMA"] = ARIMA_Model(df_temp,nobs=nobs)
                model_name.append(f"{item}_store{number}_ARIMA")

                predict_accuracy[f"{item}_Store{number}_ARIMAX"], test[f"{item}_store{number}_ARIMAX"] = ARIMA_Model(df_temp,nobs=nobs,exog_features=['Holiday','onpromotion'])
                model_name.append(f"{item}_store{number}_ARIMAX")
                predict_accuracy[f"{item}_Store{number}_SARIMA"], test[f"{item}_store{number}_SARIMA"] = SARIMA_Model(df_temp,nobs=nobs)
                model_name.append(f"{item}_store{number}_SARIMA")
                predict_accuracy[f"{item}_Store{number}_SARIMAX"], test[f"{item}_store{number}_SARIMAX"] = SARIMA_Model(df_temp,nobs=nobs,exog_features=['Holiday','onpromotion'])
                model_name.append(f"{item}_store{number}_SARIMAX")
            
            TS_chart(test,name=f"{item}_Store{number}_TimeSeries_chart",models=model_name)
            
    return predict_accuracy
            
def main():
    kaggle_project_name = "store-sales-time-series-forecasting"
    download_competition_data(competition=kaggle_project_name)
    load_kaggle_data(file_name=kaggle_project_name)
    path = os.getcwd()
    df, df_hol, df_oil, df_txn = data_loading(path)
    df = Holiday(df,df_hol)
    family_items = ['GROCERY I', 'GROCERY II', 'hardware']
    store_number = [1,2,3]
    predict_accuracy = select_model(df,df_oil,nobs=15,family=family_items,store=store_number)
    print(predict_accuracy)   


if __name__ =='__main__':
    main()
