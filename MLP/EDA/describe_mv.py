import pandas as pd
import matplotlib.pyplot as plt
import os

def makeFile(output_path, dataFrame, fileName):
    if os.path.exists(output_path) == True:
        pass
    else:
        os.mkdir(output_path)
    
    dataFrame.to_csv(path_or_buf=output_path + '\\' + fileName, encoding='cp949')

def mv_describe(dataFrame, num):
    dataFrame = dataFrame.iloc[:, 1 : dataFrame.shape[num] : 5]
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

    dataFrame2 = mv_describe(dataFrame=dataFrame, num=0)
    dataFrame3 = mv_describe(dataFrame=dataFrame, num=1)
    
    #측정값을 describe한 데이터프레임 csv 파일 생성
    mvDescribe = dataFrame2.describe()
    makeFile(outputPath, mvDescribe, "측정값_describe.csv")

    shape = dataFrame2.shape[1]
    print(shape)
    lambda_func = lambda shape: 1 if shape % 8 != 0 else 0
    addition_count = lambda_func(shape)
    repeat_count = int(shape / 8) + addition_count
    print(repeat_count)
    for i in range(0, shape, repeat_count):
        fig, axes = plt.subplots(1, 8, figsize=(15,8))
        for j in range(0, 8):
            if(i + j < shape):
                axes[j].hist(dataFrame2.iloc[:, i+j], alpha=0.75)
                if(i + j < shape -1):
                    axes[j].axvline(x=dataFrame3.iloc[0, i+j], color='r', label='a')
        plt.show()

    print(dataFrame2[dataFrame2.iloc[:, 5] == 0])
    