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

    #판정_원5(I)_Y는 대부분의 파일에 존재하지 않음 그러므로 열 삭제
    if "판정_원5(I)_Y" in dataFrame.columns or "판정_원5(I)_D" in dataFrame.columns:
        dataFrame = dataFrame.drop(columns=['판정_원5(I)_Y', '판정_원5(I)_D'])

    #두번 측정해서 열이 평소보다 많은 데이터를 두번째 측정한 데이터를 사용하도록 만듦
    if dataFrame.shape[1] > 63:
        dataFrame = dataFrame.iloc[:, 62:dataFrame.shape[1]]
    
    dataFrame.columns = [col.replace("#2", "") for col in dataFrame.columns]

    #'소재'라는 문자열이 들어간 열 제거
    dataFrame = dataFrame.loc[:, ~dataFrame.columns.str.contains('소재')]
    
    return dataFrame

def DFtoModifiedDF(dataFrame, fileName, labels):
    data = drop_null_judgement(dataFrame)
    data = modify_shape_name(data)
    data = modify_quality3(data)
    data = modify_judgement(labels=labels, dataFrame=data, file=fileName)
    data = data.fillna(value=0)
    
    return data

if __name__=="__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('mode.chained_assignment',  None)

    dataPath = os.getcwd() + "\\csv_datas_hd\\45926-4G100"
    dataList = os.listdir(dataPath)

    labels = ['----|', '---|', '--|', '-|', '>|<', '|+', '|++', '|+++', '|++++']

    dataFrame = pd.DataFrame()
    for index, file in enumerate(dataList):
        print(f"index: {index}, file: {file}")
        data = pd.read_csv(dataPath + "\\" + file, encoding='cp949')
        data = DFtoModifiedDF(dataFrame=data, fileName=file, labels=labels)
        if data.loc[:,'품질상태'].iloc[0] != 2:
            dataFrame = pd.concat([dataFrame, data], ignore_index=False)

    print(dataFrame.isnull().sum())
    
    output_path = os.getcwd() + "\\MLP&ML\\datas"
    if os.path.exists(output_path) != True:
        os.mkdir(output_path)
    
    dataFrame.to_csv(path_or_buf=output_path + '\\' + "data_jd_hd_no_NTC.csv", encoding='cp949')