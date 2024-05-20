import pandas as pd
import matplotlib as plt
import csv
import os

def makeFile(output_path):
    if os.path.exists(output_path) == True:
        pass
    else:
        os.mkdir(output_path)
    
    dataFrame.to_csv(path_or_buf=output_path + '\\' + "data_analysis.csv", encoding='cp949')

if __name__=="__main__":
    datasetPath = os.getcwd() + "\\dataset_csv\\45926-4G100"
    #datasetPath = os.getcwd() + "\\output_test_ld\\45926-4G100"
    datalist = os.listdir(datasetPath)

    dataFrame = pd.DataFrame(columns=['fileName', 'NG_Count', '품질'])
    for file in datalist:
        data = pd.read_csv(datasetPath + "\\" + file, encoding='cp949')
        datas = pd.DataFrame(data)
        numeric_values = pd.to_numeric(datas['판정'], errors='coerce')
        count_NG = pd.notnull(numeric_values).sum()
        quality = datas['품질상태'][0]
        if(quality == 'OK'):
            quality = 1
        elif(quality == 'NG'):
            quality = 0
        else:
            quality = 2
        dataFrame = pd.concat([dataFrame, pd.DataFrame({"fileName": [file], "NG_Count": [count_NG], "품질": [quality] })], ignore_index=False)
    dataFrame = dataFrame.reset_index()
    dataFrame.drop(columns=['index'], inplace=True)

    output_path = os.getcwd() + "\\MLP\\test"
    #makeFile(output_path)
    labels = dataFrame['품질'].value_counts()
    print(labels)
    
    NGs = dataFrame['NG_Count'].value_counts()
    print(NGs)
    
    print(dataFrame['NG_Count'].describe())

    print(dataFrame['NG_Count'][dataFrame['품질'] == 1].value_counts())
    print(dataFrame['NG_Count'][dataFrame['품질'] == 0].value_counts())
    print(dataFrame['NG_Count'][dataFrame['품질'] == 2].value_counts())
    print(dataFrame[(dataFrame['품질'] == 0) & (dataFrame['NG_Count'] == 3)])
    