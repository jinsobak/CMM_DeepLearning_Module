import pandas as pd
import numpy as np
import os
import csv

def fill_null_value_dv_mv_sv(dataFrame):
    print(dataFrame.shape[1])
    for i in range(1, dataFrame.shape[1]-1, 3):
        print(f"set: {i}, {i+1}, {i+2}")
        #print(dataFrame.iloc[:,i:i+1].isnull().sum())
        #print(dataFrame.iloc[:,i:i+1].median())
        print(dataFrame.iloc[:,i].dtypes)
        print(dataFrame.iloc[:,i+1].dtypes)
        print(dataFrame.iloc[:,i+2].dtypes)
        dataFrame.iloc[:,i] = dataFrame.iloc[:,i].fillna(dataFrame.iloc[:, i].median())
        dataFrame.iloc[:,i+1] = dataFrame.iloc[:,i+1].fillna(dataFrame.iloc[:,i+1].median())
        dataFrame.iloc[:, i+2] = abs(dataFrame.iloc[:, i]) - abs(dataFrame.iloc[:, i+1])

    return dataFrame

def fill_null_value_dv_mv_sv_ut_lt(dataFrame):
    print(dataFrame.shape[1])
    for i in range(1, dataFrame.shape[1]-1, 5):
        print(f"set: {i}, {i+1}, {i+2}, {i+3}, {i+4}")
        #print(dataFrame.iloc[:,i:i+1].isnull().sum())
        #print(dataFrame.iloc[:,i:i+1].median())
        print(dataFrame.iloc[:,i:i+1].dtypes)
        print(dataFrame.iloc[:,i+1:i+2].dtypes)
        print(dataFrame.iloc[:,i+2:i+3].dtypes)
        dataFrame.iloc[:,i] = dataFrame.iloc[:,i].fillna(dataFrame.iloc[:, i].median())
        dataFrame.iloc[:,i+1] = dataFrame.iloc[:,i+1].fillna(dataFrame.iloc[:,i+1].median())
        dataFrame.iloc[:,i+2] = dataFrame.iloc[:,i+2].fillna(dataFrame.iloc[:,i+2].median())
        dataFrame.iloc[:,i+3] = dataFrame.iloc[:,i+3].fillna(dataFrame.iloc[:,i+3].median())
        dataFrame.iloc[:, i+4] = abs(dataFrame.iloc[:, i]) - abs(dataFrame.iloc[:, i+1])

    return dataFrame

if __name__=="__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    dataPath = os.getcwd() + "\\MLP&ML\\datas\\data_mv_sv_dv_ut_lt_hd.csv"
    data = pd.read_csv(filepath_or_buffer=dataPath, encoding='cp949')
    dataFrame = pd.DataFrame(data)

    #dataFrame = fill_null_value_dv_mv_sv(dataFrame)
    dataFrame = fill_null_value_dv_mv_sv_ut_lt(dataFrame)
    print(dataFrame.isnull().sum())

    output_path = os.getcwd() + "\\MLP&ML\\datas"
    if os.path.exists(output_path) == True:
        pass
    else:
        os.mkdir(output_path)
    
    #fileName = "data_mv,sv,dv_hd_test.csv"
    fileName = "data_mv_sv_dv_ut_lt_hd_test.csv"

    dataFrame.to_csv(path_or_buf=output_path + '\\' + fileName, encoding='cp949')
