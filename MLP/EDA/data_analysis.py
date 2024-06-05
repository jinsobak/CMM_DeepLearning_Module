import pandas as pd
import matplotlib as plt
import csv
import os

def makeFile(output_path):
    if os.path.exists(output_path) == True:
        pass
    else:
        os.mkdir(output_path)
    
    dataFrame2.to_csv(path_or_buf=output_path + '\\' + "data_analysis.csv", encoding='cp949')

if __name__=="__main__":
    datasetPath = os.getcwd() + "\\MLP\\datas\\data_jd_hd.csv"
    dataFrame = pd.read_csv(datasetPath, encoding='cp949')

    dataFrame_fileName = dataFrame['파일명']
    dataFrame = dataFrame.drop(columns=['파일명'])

    dataFrame_quality = dataFrame['품질상태']
    dataFrame = dataFrame.drop(columns=['품질상태'])

    dataFrame2 = pd.DataFrame(columns=['fileName', 'NG_Count', '품질'])
    for i in range(0, dataFrame.shape[0]):
        count = 0
        for j in range(0, dataFrame.shape[1]):
            if dataFrame.iloc[i , j] != 0:
                count += 1
        dataFrame2 = pd.concat([dataFrame2, pd.DataFrame({"fileName": [dataFrame_fileName[i]], "NG_Count": [count], "품질": [dataFrame_quality[i]] })], ignore_index=False)
    dataFrame2 = dataFrame2.reset_index()
    dataFrame2.drop(columns=['index'], inplace=True)

    output_path = os.getcwd() + "\\MLP\\test"
    #makeFile(output_path)
    labels = dataFrame2['품질'].value_counts()
    print(labels)
    
    NGs = dataFrame2['NG_Count'].value_counts()
    print(NGs)
    
    print(dataFrame2['NG_Count'].describe())

    print(dataFrame2['NG_Count'][dataFrame2['품질'] == 1].value_counts())
    print(dataFrame2['NG_Count'][dataFrame2['품질'] == 0].value_counts())
    print(dataFrame2['NG_Count'][dataFrame2['품질'] == 2].value_counts())
    print(dataFrame2[(dataFrame2['품질'] == 0) & (dataFrame2['NG_Count'] == 3)])
    