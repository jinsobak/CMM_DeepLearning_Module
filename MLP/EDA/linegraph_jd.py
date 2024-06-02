import pandas as pd
import matplotlib.pyplot as plt
import os
import describe_mv as des

if __name__=="__main__":
    plt.rcParams['font.family'] ='Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] =False
    #출처: https://giveme-happyending.tistory.com/168 [소연의_개발일지:티스토리]

    dataPath = os.getcwd() + "\\MLP\\datas\\data_jd_hd.csv"
    outputPath = os.getcwd() + "\\MLP\\EDA\\datas\\"

    dataFrame = pd.read_csv(dataPath, encoding='cp949')
    print(dataFrame.shape)
    print(dataFrame.head())
    print(dataFrame.dtypes)

    dataFrame2 = dataFrame.loc[dataFrame['품질상태'] == 0, :]
    dataFrame3 = dataFrame.loc[dataFrame['품질상태'] == 1, :]
    dataFrame4 = dataFrame.loc[dataFrame['품질상태'] == 2, :]

    dataFrame2_fileName = dataFrame2.iloc[:, 0]
    print(dataFrame2_fileName.head())
    dataFrame2 = dataFrame2.iloc[:, 1:]
    dataFrame3 = dataFrame3.iloc[:, 1:]
    dataFrame4 = dataFrame4.iloc[:, 1:]

    dataFrame2 = dataFrame2.loc[~((dataFrame2['판정_점2 <- 점1의 되부름 <열전 관리치수(Spec : 116.6±0.1)>_X'] != 0) | (dataFrame2['판정_직선21 <우하 소재>_X/Y'] != 0))]
    print(dataFrame2.shape[0])

    ylim_under = -0.3
    ylim_over = 0.3
    
    plt.figure("판정 선그래프 NG", figsize=(15, 8))
    for i in range(0, dataFrame2.shape[0]):
        plt.plot(dataFrame2.iloc[i , :])
        plt.ylim(ylim_under, ylim_over)
        plt.title(f"{dataFrame2_fileName.iloc[i]} 판정 선그래프 NG")
        plt.show()

    # plt.figure("판정 선그래프 OK", figsize=(15, 8))
    # for i in range(0, dataFrame3.shape[0]):
    #     plt.plot(dataFrame3.iloc[i, :])
    #     plt.ylim(ylim_under, ylim_over)
    #     plt.title(f"판정 선그래프 OK")
    #     plt.show()

    # plt.figure("판정 선그래프 NTC", figsize=(15, 8))
    # for i in range(0, dataFrame4.shape[0]):
    #     plt.plot(dataFrame4.iloc[i, :])
    # plt.xticks(ticks=range(dataFrame4.shape[1]))
    # plt.ylim(ylim_under, ylim_over)
    # plt.title(f"판정 선그래프 NTC")
    # plt.show()