import pandas as pd
import os
import numpy as np

# 기존 함수들은 변경 없이 사용

if __name__=="__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('mode.chained_assignment', None)
    
    # 상위 폴더 경로 지정
    topDatasetPath = os.getcwd() + "\\output_test_ld"
    productList = os.listdir(topDatasetPath)  # 제품 폴더 리스트

    for product in productList:
        datasetPath = os.path.join(topDatasetPath, product)
        datalist = os.listdir(datasetPath)

        dataFrame = pd.DataFrame()
        for file in datalist:
            data = pd.read_csv(os.path.join(datasetPath, file), encoding='cp949')
            datas = pd.DataFrame(data)
            datas = fill_or_drop_deviation(datas)
            dataFrame = pd.concat([dataFrame, datas], ignore_index=True)
        
        # 데이터 증강
        dataFrame = augment_data(dataFrame, 1000)

        output_path = os.getcwd() + "\\MLP\\test\\" + product  # 각 제품별 폴더 생성
        os.makedirs(output_path, exist_ok=True)
        
        dataFrame.to_csv(path_or_buf=os.path.join(output_path, "sampling2.csv"), encoding='cp949')