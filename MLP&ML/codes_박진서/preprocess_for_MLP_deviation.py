import pandas as pd
import csv
import os
import matplotlib.pyplot as mat

def devch(datas):
    if datas['편차'] == '-' and datas['판정'] != '-':
        datas['편차'] = abs(float(datas['측정값'])) - abs(float(datas['기준값']))
    return datas['편차']

def preProcess(datas):
    datas.drop(datas[datas['항목'] == 'SMmf'].index, inplace=True)

    datas = datas[['품명', '도형', '항목', '측정값', '기준값','편차', '판정', '품질상태']]
    datas['편차'] = datas.apply(devch, axis = 1)
    datas.drop(datas[datas['판정']=='-'].index, inplace=True)

    return datas

def modify_quality(datas, mode):
    quality = datas["품질상태"][0]
    if mode == 1:
        if(quality == "OK"):
            quality = 1
        else:
            quality = 0
    elif mode == 2:
        if(quality == "OK"):
            quality = 1
        elif(quality == "NG"):
            quality = 0
        else:
            quality = 2

    return quality

def preProcess2(datas, mode, fileName):
    quality = modify_quality(datas, mode)
    name = datas['품명'][0]

    datas['도형'] = '편차_' + datas['도형'] + "_" + datas['항목']
    datas = datas.set_index('도형')
    datas = datas[['편차']]

    datas_tp = datas.transpose()
    datas_tp = datas_tp.reset_index()
    datas_tp = datas_tp.drop('index', axis=1)

    datas_tp.insert(loc=0, column='파일명', value=fileName)
    datas_tp = datas_tp.assign(품질상태 = quality)

    datas_tp = datas_tp.set_index('파일명')
    datas_tp.index.name = '파일명'

    return datas_tp

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('mode.chained_assignment',  None)

    mode = 2
    dataset_path = os.getcwd() + "\\csv_datas_hd\\45926-4G100"
    data_list = os.listdir(dataset_path)

    dataFrame = pd.DataFrame()
    for file in data_list:
        data = pd.read_csv(dataset_path + "\\" + file, encoding='cp949')
        datas = pd.DataFrame(data)  
        datas = preProcess(datas)
        datas = preProcess2(datas, mode, fileName=file)
        dataFrame = pd.concat([dataFrame, datas], ignore_index=False)

    for col in dataFrame.columns:
        nanRatio = dataFrame[col].isnull().sum() / dataFrame[col].shape[0]
        #print(col + " " + str(nanRatio))
        if(nanRatio >= 0.5):
            dataFrame.drop(columns=[col], inplace=True)

    dataFrame.fillna(value=0, inplace=True)

    output_path = os.getcwd() + "\\MLP&ML\\datas"
    if os.path.exists(output_path) == True:
        pass
    else:
        os.mkdir(output_path)
    
    if mode == 1:
        dataFrame.to_csv(path_or_buf=output_path + '\\' + "data_dv_hd.csv", encoding='cp949')
    elif mode == 2:
        dataFrame.to_csv(path_or_buf=output_path + '\\' + "data_dv_hd_with_NTC.csv", encoding='cp949')

    