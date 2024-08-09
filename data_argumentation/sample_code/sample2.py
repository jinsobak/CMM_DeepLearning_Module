import pandas as pd
import os
import numpy as np


if __name__=="__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('mode.chained_assignment', None)
    

    topDatasetPath = os.getcwd() + "\\output_test_ld"
    productList = os.listdir(topDatasetPath)  

    for product in productList:
        datasetPath = os.path.join(topDatasetPath, product)
        datalist = os.listdir(datasetPath)

        dataFrame = pd.DataFrame()
        for file in datalist:
            data = pd.read_csv(os.path.join(datasetPath, file), encoding='cp949')
            datas = pd.DataFrame(data)
            datas = fill_or_drop_deviation(datas)
            dataFrame = pd.concat([dataFrame, datas], ignore_index=True)
        
  
        dataFrame = augment_data(dataFrame, 1000)

        output_path = os.getcwd() + "\\MLP\\test\\" + product  
        os.makedirs(output_path, exist_ok=True)
        
        dataFrame.to_csv(path_or_buf=os.path.join(output_path, "sampling2.csv"), encoding='cp949')