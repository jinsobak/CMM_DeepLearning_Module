import pandas as pd
import matplotlib as plt
import csv
import os

if __name__=="__main__":
    datasetPath = os.getcwd() + "\\output_test_ld\\45926-4G100"
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
        else:
            quality = 0
        dataFrame = pd.concat([dataFrame, pd.DataFrame({"fileName": [file], "NG_Count": [count_NG], "품질": [quality] })], ignore_index=False)
    


    output_path = os.getcwd() + "\\MLP\\test"
    if os.path.exists(output_path) == True:
        pass
    else:
        os.mkdir(output_path)
    
    dataFrame.to_csv(path_or_buf=output_path + '\\' + "data_analysis.csv", encoding='cp949')