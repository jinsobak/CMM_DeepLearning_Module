import pandas as pd
import numpy as np
import os
import csv

def fill_null_value(dataFrame):
    print(dataFrame.shape[1])
    for i in range(1, dataFrame.shape[1]-1, 3):
        print(f"set: {i}, {i+1}, {i+2}")
        #print(dataFrame.iloc[:,i:i+1].isnull().sum())
        #print(dataFrame.iloc[:,i:i+1].median())
        print(dataFrame.iloc[:,i:i+1].dtypes)
        print(dataFrame.iloc[:,i+1:i+2].dtypes)
        print(dataFrame.iloc[:,i+2:i+3].dtypes)
        dataFrame.iloc[:,i:i+1] = dataFrame.iloc[:,i:i+1].fillna(dataFrame.iloc[:, i:i+1].median())
        dataFrame.iloc[:,i+1:i+2] = dataFrame.iloc[:,i+1:i+2].fillna(dataFrame.iloc[:,i+1:i+2].median())
        dataFrame.iloc[:, i+2] = dataFrame.iloc[:, i] - dataFrame.iloc[:, i+1]

    return dataFrame

if __name__=="__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    dataPath = os.getcwd() + "\\MLP\\datas\\data_mv,sv,dv_hd.csv"
    data = pd.read_csv(filepath_or_buffer=dataPath, encoding='cp949')
    dataFrame = pd.DataFrame(data)

    dataFrame = fill_null_value(dataFrame)
    print(dataFrame.isnull().sum())

    output_path = os.getcwd() + "\\MLP\\datas"
    if os.path.exists(output_path) == True:
        pass
    else:
        os.mkdir(output_path)
    
    dataFrame.to_csv(path_or_buf=output_path + '\\' + "data_mv,sv,dv_hd_test.csv", encoding='cp949')
