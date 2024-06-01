import pandas as pd
import matplotlib as plt
import os
import csv

if __name__=="__main__":
    filePath = os.getcwd() + "\\MLP\\datas\\data_mv,sv,dv_hd_with_NTC.csv"

    dataFrame = pd.read_csv(filePath, encoding='cp949')

    print(dataFrame.head())

    quality = 2
    dataFrame2 = dataFrame.loc[dataFrame['품질상태'] == quality, :]
    print(dataFrame2.head())

    if quality == 0:
        quality = 'NG'
    elif quality == 1:
        quality = 'OK'
    else:
        quality = 'NTC'

    output_path = os.getcwd() + "\\MLP\\datas"
    if os.path.exists(output_path) == True:
        pass
    else:
        os.mkdir(output_path)
    
    dataFrame2.to_csv(path_or_buf=output_path + '\\' + "data_mv_sv_dv_ut_lt_hd_only_" + str(quality) + ".csv", encoding='cp949')