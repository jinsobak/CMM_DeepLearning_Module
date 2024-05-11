import pandas as pd
import numpy as np
import os

def drop_null_judgement(dataFrame):
    dataFrame.drop(dataFrame[dataFrame['판정'] == '-'].index, inplace=True)
    
    return dataFrame

def modify_quality(dataFrame):
    quality = dataFrame['품질상태'][0]
    if(quality == "OK"):
        quality = 1
    else:
        quality = 0

    return quality

def modify_shape_name(dataFrame):
    dataFrame['번호'] = dataFrame['번호'].astype(str)
    dataFrame['도형'] = '판정_' + dataFrame['번호'] + '_' + dataFrame['도형'] + '_' + dataFrame['항목']
    dataFrame['번호'] = dataFrame['번호'].astype(int)

    return dataFrame

def ont_hot_encoder(labels, dataFrame):
    

if __name__=="__main__":
    dataPath = os.getcwd() + "\\dataset_csv\\45926-4G100"
    dataList = os.listdir(dataPath)

    outputPath = os.getcwd() + "\\MLP\\datas\\"
    print(os.path.exists(outputPath))
    if os.path.exists(outputPath) != True:
        os.mkdir(outputPath)

    labels = ['----l', '---l', '--l', '-l', 'l', 'l+', 'l++', 'l+++', 'l++++', 'out']

    dataFrame = pd.DataFrame()
    dataFrame = pd.read_csv(dataPath + "\\" + dataList[0], encoding='cp949')
    dataFrame = drop_null_judgement(dataFrame)
    dataFrame = modify_shape_name(dataFrame)

   
    