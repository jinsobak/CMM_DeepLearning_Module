import pandas as pd
import matplotlib.pyplot as plt
import os
import describe_mv as des

if __name__=="__main__":
    plt.rcParams['font.family'] ='Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] =False
    #출처: https://giveme-happyending.tistory.com/168 [소연의_개발일지:티스토리]

    dataPath = os.getcwd() + "\\MLP\\datas\\data_mv_sv_dv_ut_lt_hd_with_NTC.csv"
    outputPath = os.getcwd() + "\\MLP\\EDA\\datas\\"

    dataFrame = pd.read_csv(dataPath, encoding='cp949')
    print(dataFrame.shape)
    print(dataFrame.head())

    qualities = dataFrame['품질상태']
    dataFrame2 = des.mv_describe(dataFrame=dataFrame, num=5, label_input='0')
    print(dataFrame2.dtypes)
    dataFrame3 = des.mv_describe(dataFrame=dataFrame, num=5, label_input='1')
    dataFrame4 = des.mv_describe(dataFrame=dataFrame, num=5, label_input='2')
    dataFrame5 = des.mv_describe(dataFrame=dataFrame, num=5, label_input='a')

    plt.figure("편차 선그래프 NG", figsize=(15, 8))
    for i in range(0, dataFrame2.shape[0]):
        plt.plot(dataFrame2.iloc[i, :])
    plt.xticks(ticks=range(dataFrame2.shape[1]))
    plt.ylim(-30, 40)
    plt.title(f"편차 선그래프 NG")
    plt.show()

    plt.figure("편차 선그래프 OK", figsize=(15, 8))
    for i in range(0, dataFrame3.shape[0]):
        plt.plot(dataFrame3.iloc[i, :])
    plt.xticks(ticks=range(dataFrame3.shape[1]))
    plt.ylim(-30, 40)
    plt.title(f"편차 선그래프 OK")
    plt.show()

    plt.figure("편차 선그래프 NTC", figsize=(15, 8))
    for i in range(0, dataFrame4.shape[0]):
        plt.plot(dataFrame4.iloc[i, :])
    plt.xticks(ticks=range(dataFrame4.shape[1]))
    plt.ylim(-30, 40)
    plt.title(f"편차 선그래프 NTC")
    plt.show()