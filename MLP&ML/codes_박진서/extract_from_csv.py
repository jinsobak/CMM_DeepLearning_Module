import pandas as pd
import matplotlib as plt
import os
import csv

if __name__=="__main__":
    filePath = os.getcwd() + "\\MLP&ML\\datas\\data_mv_sv_dv_ut_lt_hd_with_NTC.csv"

    dataFrame = pd.read_csv(filePath, encoding='cp949')

    print(dataFrame.head())

    quality = 2
    dataFrame2 = dataFrame.loc[(dataFrame['품질상태'] == quality), :]
    print(dataFrame2.head())

    if quality == 0:
        quality = 'NG'
    elif quality == 1:
        quality = 'OK'
    else:
        quality = 'NTC'

    output_path = os.getcwd() + "\\MLP&ML\\datas"
    if os.path.exists(output_path) == True:
        pass
    else:
        os.mkdir(output_path)
    
    dataFrame2 = dataFrame2.set_index('파일명')
    dataFrame2.to_csv(path_or_buf=output_path + '\\' + "data_mv_sv_dv_ut_lt_hd_only_" + str(quality) + ".csv", encoding='cp949')
    #print(dataFrame2[dataFrame2['판정_점2 <- 점1의 되부름 <열전 관리치수(Spec : 116.6±0.1)>_X'] != 0])