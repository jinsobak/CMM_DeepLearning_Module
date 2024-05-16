import pandas as pd
import matplotlib.pyplot as plt
import os

def makeFile(output_path, dataFrame, fileName):
    if os.path.exists(output_path) == True:
        pass
    else:
        os.mkdir(output_path)
    
    dataFrame.to_csv(path_or_buf=output_path + '\\' + fileName, encoding='cp949')

def mv_describe(dataFrame):
    dataFrame = dataFrame.iloc[:, 1 : dataFrame.shape[0] : 5]
    print(dataFrame.head())

    return dataFrame

if __name__=="__main__":
    plt.rcParams['font.family'] ='Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] =False
    #출처: https://giveme-happyending.tistory.com/168 [소연의_개발일지:티스토리]
    dataPath = os.getcwd() + "\\MLP\\datas\\data_mv_sv_dv_ut_lt_hd.csv"
    outputPath = os.getcwd() + "\\MLP\\EDA\\datas\\"

    dataFrame = pd.read_csv(dataPath, encoding='cp949')
    print(dataFrame.head())

    dataFrame2 = mv_describe(dataFrame=dataFrame)
    
    #측정값을 describe한 데이터프레임 csv 파일 생성
    dataFrameDescribe = dataFrame2.describe()
    makeFile(outputPath, dataFrameDescribe, "측정값_describe.csv")

    shape = dataFrame2.shape[1]
    print(shape)
    lambda_func = lambda shape: 1 if shape % 8 != 0 else 0
    addition_count = lambda_func(shape)
    repeat_count = int(shape / 8) + addition_count
    print(repeat_count)
    for i in range(0, shape, repeat_count):
        dataFrame2.iloc[:, i: i+8].hist(figsize=(15, 10), histtype='bar')
        plt.show()

    print(dataFrame2[dataFrame2.iloc[:, 5] == 0])
    