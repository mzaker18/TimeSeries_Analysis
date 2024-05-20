import logging as log
import datetime
import kaggle
import pandas as pd
import numpy as np
import json
import os
import subprocess
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns

def download_competition_data(competition='store-sales-time-series-forecasting'):
    '''
    download whole datasets related to a project from kaggle
    '''
    print(f"Strat to download {competition} datasets as {competition}.zip")
    result = subprocess.run(['kaggle', 'competitions', 'download', '-c', competition], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Downloaded data for competition: {competition}")
    else:
        print(f"Failed to download data: {result.stderr}")

    print(f"Finished downloading {competition} datasets as {competition}.zip")


def load_kaggle_data(file_name:str):
    '''
    Download and save datasets in a data folder
    Input file_name: comptetition file_name in Kaggle
    '''
    print(f"Strat to save {file_name} datasets in data folder")
    if os.path.isdir(f"{os.getcwd()}//data")==False:
        os.mkdir(f"{os.getcwd()}//data")
    with zipfile.ZipFile(f"{os.getcwd()}//{file_name}.zip",'r') as zf:
        for file in zf.namelist():
            if file.split('.')[1]=='csv':
                data = pd.read_csv(zf.open(file))
                data.to_csv(f"{os.getcwd()}//data//{file.split('.')[0]}.csv",index=True)

    print(f"Finish saving {file_name}datasets in data folder")


def data_loading(path=os.getcwd()):
    '''
    This function is used to load dataset
    Input: Path point to dataset directory (future will include direct API call from kaggle)
    Output: sales dataset merged with store information (df), holiday dataset (df_hol), transaction dataset (df_txn), daily oil price (df_oil) 
    '''
    df_txn = pd.read_csv(f'{path}//data//transactions.csv')
    df_oil = pd.read_csv(f'{path}//data//oil.csv')
    df_str = pd.read_csv(f'{path}//data//stores.csv')
    df_hol = pd.read_csv(f'{path}//data//holidays_events.csv')
    df_train = pd.read_csv(f'{path}//data//train.csv')
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_txn['date'] = pd.to_datetime(df_txn['date'])
    df_oil['date'] = pd.to_datetime(df_oil['date'])
    df_hol['date'] = pd.to_datetime(df_hol['date'])
    
    #Manipulate missing values in oil price on weekend and holidays: by replacing oil_price on previous day
    for index, row in df_oil.iterrows():
        if pd.isna(row['dcoilwtico']):
            if 0 < index < len(df_oil)-1:
                df_oil.at[index,'dcoilwtico']=df_oil['dcoilwtico'].iloc[index-1]
            elif index == 0:
                df_oil.at[index,'dcoilwtico']=df_oil['dcoilwtico'].iloc[index+1]
            elif index == len(df_oil)-1:
                df_oil.at[index,'dcoilwtico']=df_oil['dcoilwtico'].iloc[index-1]
    df_oil.rename(columns={'dcoilwtico':'oil_price'},inplace=True)
    
    df = df_train.merge(df_str,on='store_nbr',how='left')
    
    return df, df_hol, df_oil, df_txn


def Holiday(df: pd.DataFrame , df_hol: pd.DataFrame):
    '''
    This function consider final dataframe (df) and raw information about holidays and then based on date
    of sales and store information such as city, state will label if the sales happen on holiday or not.
    Input df: Sales Time Series 
    Input df_hol: Holiday information based on city, state and date
    Output df: Updated time series data with holiday information
    '''
    # First step: Consider national holiday dates which apply to all sales, regardless of store location.
    df_hol_national = df_hol[df_hol['locale'].str.lower()=='national']
    df_hol_national.drop_duplicates(subset='date',keep='last',inplace=True)
    df_hol_national.reset_index(inplace=True,drop=True)
    
    # Second step: consider regional holiday dates which apply to sales in states which are holiday.
    df_hol_reg = df_hol[df_hol['locale']=='Regional']
    df_hol_reg.drop_duplicates(subset=['date','locale_name'],keep='last',inplace=True)
    df_hol_reg.rename(columns={"locale_name":"state"},inplace=True)
    
    # Third step: consider local holiday dates which apply to sales in cities which are holiday.
    df_hol_local = df_hol[df_hol['locale'].str.lower()=='local']
    df_hol_local.drop_duplicates(subset=['date','locale_name'],keep='last',inplace=True)
    df_hol_local.reset_index(inplace=True,drop=True)
    df_hol_local.rename(columns={"locale_name":"city"},inplace=True)


    #Using update function alongside merge function
    df_temp1 = df.copy()
    df_temp1 = df_temp1.merge(df_hol_national,on='date',how='left')
    df_temp2= df.copy()
    df_temp2 = df_temp2.merge(df_hol_local,on=['date','city'],how='left')
    df_temp3 = df.copy()
    df_temp3 = df_temp3.merge(df_hol_reg,on=['date','state'],how='left')
    
    #set index to a combination of 'id' and 'date' to update respective rows in temp1 based on temp2 and temp3
    df_temp1.set_index(['id', 'date'], inplace=True)
    df_temp2.set_index(['id', 'date'], inplace=True)
    df_temp3.set_index(['id', 'date'], inplace=True)
    df_temp1.update(df_temp2)
    df_temp1.update(df_temp3)
    
    #Reset index to normal and return prepared dataset
    df = df_temp1.reset_index()
    df['Holiday'] = df.apply(lambda row: 'True' if (pd.notnull(row['transferred']) and (row['transferred'] == False)) else 'False', axis=1)
    
    #drop undesired columns
    df.drop(columns=['description', 'transferred', 'locale', 'locale_name', 'type_y'],inplace=True)
    
    return df

def TS_chart(test: pd.DataFrame,name: str,models:list,path=f"{os.getcwd()}//plots"):
    '''
    Creating time series chart of real sales and predicted values by different models.
    Input pred_df: time series prediction from different models
    Input name: generic name to save chart
    Input path: using this path to save plots
    '''
    test = test[models]
    test.plot(figsize=(12,8),legend=True)
    plt.title(f"Sales Time Series for {name}")
    if os.path.isdir(path)==False:
        os.makedirs(path)
    plt.savefig(f'{path}//{name}.png')
