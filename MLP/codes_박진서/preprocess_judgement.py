import pandas as pd
import numpy as np
import os

def drop_null_judgement(dataFrame):
    dataFrame.drop(dataFrame[dataFrame['판정'] == '-'].index, inplace=True)
    
    return dataFrame

def modify_quality3(dataFrame):
    quality = dataFrame['품질상태'][0]
    if(quality == "OK"):
        quality = 1
    elif(quality == "NG"):
        quality = 0
    else:
        quality = 2
    dataFrame['품질상태'] = quality

    return dataFrame

def modify_quality2(dataFrame):
    quality = dataFrame['품질상태'][0]
    if(quality == "OK"):
        quality = 1
    else:
        quality = 0
    dataFrame['품질상태'] = quality

    return dataFrame

def modify_shape_name(dataFrame):
    dataFrame['도형'] = '판정_' + dataFrame['도형'] + '_' + dataFrame['항목']

    return dataFrame

def modify_judgement(labels, dataFrame, file):
    name = file
    quality = dataFrame['품질상태'][0]
    dataFrame = dataFrame.assign(판정2=dataFrame['판정'])
    dataFrame.loc[dataFrame['판정'].isin(labels), '판정2'] = 0

    dataFrame = dataFrame.set_index('도형')
    dataFrame = dataFrame[['판정2']].transpose()
    dataFrame = dataFrame.reset_index()
    dataFrame = dataFrame.drop('index', axis=1)
    dataFrame = dataFrame.astype(dtype='float64')
    dataFrame.insert(0, '파일명', name)
    dataFrame = dataFrame.assign(품질상태=quality)
    dataFrame = dataFrame.set_index('파일명')
    dataFrame.index.name = '파일명'

    return dataFrame

if __name__=="__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('mode.chained_assignment',  None)

    dataPath = os.getcwd() + "\\dataset_csv\\45926-4G100"
    dataList = os.listdir(dataPath)

    labels = ['----|', '---|', '--|', '-|', '>|<', '|+', '|++', '|+++', '|++++']

    dataFrame = pd.DataFrame()
    for index, file in enumerate(dataList):
        print(f"index: {index}, file: {file}")
        data = pd.read_csv(dataPath + "\\" + file, encoding='cp949')
        data = drop_null_judgement(data)
        data = modify_shape_name(data)
        #data = modify_quality3(data)
        data = modify_quality2(data)
        data = modify_judgement(labels=labels, dataFrame=data, file=file)
        dataFrame = pd.concat([dataFrame, data], ignore_index=False)

    for col in dataFrame.columns:
        nanRatio = dataFrame[col].isnull().sum() / dataFrame[col].shape[0]
        #print(col + " " + str(nanRatio))
        if(nanRatio >= 0.5):
            dataFrame.drop(columns=[col], inplace=True)

    dataFrame = dataFrame.fillna(value=0)
    print(dataFrame.isnull().sum())

    output_path = os.getcwd() + "\\MLP\\datas"
    if os.path.exists(output_path) != True:
        os.mkdir(output_path)
    
    dataFrame.to_csv(path_or_buf=output_path + '\\' + "data_jd_2_hd.csv", encoding='cp949')