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

    dataFrame2 = dataFrame2.iloc[:, 1:]
    dataFrame3 = dataFrame3.iloc[:, 1:]
    dataFrame4 = dataFrame4.iloc[:, 1:]

    min_max_dataFrame = pd.DataFrame()

    dataFrame2_max = dataFrame2.max()
    print(dataFrame2_max)
    min_max_dataFrame = pd.concat([min_max_dataFrame, dataFrame2_max], axis=1)
    dataFrame2_min = dataFrame2.min()
    min_max_dataFrame = pd.concat([min_max_dataFrame, dataFrame2_min], axis=1)

    dataFrame3_max = dataFrame3.max()
    print(dataFrame3_max)
    min_max_dataFrame = pd.concat([min_max_dataFrame, dataFrame3_max], axis=1)
    dataFrame3_min = dataFrame3.min()
    min_max_dataFrame = pd.concat([min_max_dataFrame, dataFrame3_min], axis=1)

    dataFrame4_max = dataFrame4.max()
    print(dataFrame4_max)
    min_max_dataFrame = pd.concat([min_max_dataFrame, dataFrame4_max], axis=1)
    dataFrame4_min = dataFrame4.min()
    min_max_dataFrame = pd.concat([min_max_dataFrame, dataFrame4_min], axis=1)

    print(min_max_dataFrame)

    makeFileInput = input("csv파일을 생성하시겠습니까?(y, n): ")
    if makeFileInput == 'y':
        des.makeFile(outputPath, min_max_dataFrame, "판정_min_max_with_NTC.csv", "")

    ylim_under = -1
    ylim_over = 1

    # plt.figure("판정 산점도 NG", figsize=(15, 8))
    # for i in range(0, dataFrame2.shape[0]):
    #     plt.scatter(dataFrame2.columns, dataFrame2.iloc[i, :])
    # plt.ylim(ylim_under, ylim_over)
    # plt.title(f"판정 산점도 NG")
    # plt.show()

    # plt.figure("판정 산점도 OK", figsize=(15, 8))
    # for i in range(0, dataFrame3.shape[0]):
    #     plt.scatter(dataFrame3.columns, dataFrame3.iloc[i, :])
    # plt.ylim(ylim_under, ylim_over)
    # plt.title(f"판정 산점도 OK")
    # plt.show()

    # plt.figure("판정 산점도 NTC", figsize=(15, 8))
    # for i in range(0, dataFrame4.shape[0]):
    #     plt.scatter(dataFrame4.columns, dataFrame4.iloc[i, :])
    # plt.ylim(ylim_under, ylim_over)
    # plt.title(f"판정 산점도 NTC")
    # plt.show()