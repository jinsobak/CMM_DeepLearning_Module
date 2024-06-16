import pandas as pd
import matplotlib.pyplot as plt
import os
import describe_mv as des

if __name__=="__main__":
    plt.rcParams['font.family'] ='Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] =False
    #출처: https://giveme-happyending.tistory.com/168 [소연의_개발일지:티스토리]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    dataPath = os.getcwd() + "\\MLP\\datas\\data_jd_hd.csv"
    outputPath = os.getcwd() + "\\MLP\\EDA\\datas\\"

    dataFrame = pd.read_csv(dataPath, encoding='cp949')
    print(dataFrame.shape)

    quality = 0
    dataFrame2 = dataFrame.loc[dataFrame['품질상태'] == quality, :]
    print(dataFrame2.shape)


    # dataFrame2 = dataFrame2.loc[~(
    #     (dataFrame2['판정_평면3 <- 점21, 점22, 점23, ... , 점26의 측정점 병합 <마이크로미터로 측정 해볼 것 !!!!!!!!!!!!!!!!>_Z'] > 0.645)
    #     & (abs(dataFrame2['판정_점2 <- 점1의 되부름 <열전 관리치수(Spec : 116.6±0.1)>_X']) > 0.007) 
    #     & (abs(dataFrame2['판정_직선21 <우하 소재>_X/Y']) > 0.07)
    # )]

    # dataFrame2 = dataFrame2.loc[~(
    #     (abs(dataFrame2['판정_점2 <- 점1의 되부름 <열전 관리치수(Spec : 116.6±0.1)>_X']) > 0.04)
    #      & (dataFrame2['판정_평면3 <- 점21, 점22, 점23, ... , 점26의 측정점 병합 <마이크로미터로 측정 해볼 것 !!!!!!!!!!!!!!!!>_Z'] > 0.645)
    # )]

    # dataFrame2 = dataFrame2.loc[(
    #     (abs(dataFrame2['판정_점13 <- 점11와 점12의 중점 <열전관리_상>_X']) == 0) 
    #     & (abs(dataFrame2['판정_점18 <- 점16와 점17의 중점 <열전관리_하>_X']) != 0)
    #     #& (dataFrame2['판정_평면3 <- 점21, 점22, 점23, ... , 점26의 측정점 병합 <마이크로미터로 측정 해볼 것 !!!!!!!!!!!!!!!!>_Z'] >= 0.6)
    # )]

    print(dataFrame2.shape)

    dataFrame2_fileName = dataFrame2.iloc[:, 0]
    dataFrame2 = dataFrame2.iloc[:, 1:]

    for i in range(0, dataFrame2.shape[1]):
        count = 0
        for j in range(0, dataFrame2.shape[0]):
            if(dataFrame2.iloc[j, i] != 0):
                count += 1
        print(f"{dataFrame2.columns[i]}: {count}")
    
    ylim_under = -0.15
    ylim_over = 0.7
    
    for i in range(0, dataFrame2.shape[0]):
        plt.figure(f"{dataFrame2_fileName.iloc[i]} 판정 선그래프", figsize=(15, 8))
        plt.plot(dataFrame2.iloc[i , :])
        plt.ylim(ylim_under, ylim_over)
        plt.title(f"{i}. {dataFrame2_fileName.iloc[i]} 판정 선그래프")
        plt.show()