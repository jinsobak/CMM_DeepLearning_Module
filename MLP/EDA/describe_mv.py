import pandas as pd
import matplotlib.pyplot as plt
import os

def mv_describe(dataFrame):
    dataFrame = dataFrame.iloc[:, 1 : dataFrame.shape[0] : 5]
    print(dataFrame.head())

    return dataFrame

if __name__=="__main__":
    plt.rcParams['font.family'] ='Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] =False
    #출처: https://giveme-happyending.tistory.com/168 [소연의_개발일지:티스토리]
    dataPath = os.getcwd() + "\\MLP\\datas\\data_mv_sv_dv_ut_lt_hd.csv"

    dataFrame = pd.read_csv(dataPath, encoding='cp949')
    print(dataFrame.head())

    dataFrame = mv_describe(dataFrame=dataFrame)
    
    shape = dataFrame.shape[1]
    print(shape)
    dataFrame.iloc[:, shape-10:shape].hist(figsize=(15, 10))
    plt.show()

    # print(dataFrame.iloc[:, 7].describe())
    # print(dataFrame.iloc[:, 7].median())
    