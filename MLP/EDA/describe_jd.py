import pandas as pd
import matplotlib.pyplot as plt
import os
import describe_mv as des

def jd_describe(dataFrame, num, label_input):
    if label_input == '0' or label_input == '1' or label_input == '2':
        label_input = int(label_input)
        dataFrame = dataFrame.loc[dataFrame['품질상태'] == label_input]
        print(dataFrame['품질상태'].head())
        dataFrame = dataFrame.iloc[:, num : dataFrame.shape[1]]
    elif label_input == 'a':
        dataFrame = dataFrame.iloc[:, num : dataFrame.shape[1]]
    print(dataFrame.head())

    return dataFrame

if __name__=="__main__":
    plt.rcParams['font.family'] ='Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] =False
    #출처: https://giveme-happyending.tistory.com/168 [소연의_개발일지:티스토리]
    
    #OK, NG, NTC만 볼지, 전부 볼지 입력
    label_input = input("표시하고 싶은 라벨을 적으세요.(0, 1, 2, a): ")
    
    dataPath = os.getcwd() + "\\MLP\\datas\\data_jd_hd.csv"
    outputPath = os.getcwd() + "\\MLP\\EDA\\datas\\"

    dataFrame = pd.read_csv(dataPath, encoding='cp949')
    print(dataFrame.shape)
    print(dataFrame.head())

    dataFrame2 = jd_describe(dataFrame=dataFrame, num=1, label_input=label_input)
    
    #측정값을 describe한 데이터프레임 csv 파일 생성
    makeFileInput = input("csv파일을 생성하시겠습니까?(y, n): ")
    if makeFileInput == 'y':
        mvDescribe = dataFrame2.describe()
        des.makeFile(outputPath, mvDescribe, "판정_describe_with_NTC.csv", label_input)