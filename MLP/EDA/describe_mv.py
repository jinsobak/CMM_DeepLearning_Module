import pandas as pd
import matplotlib.pyplot as plt
import os

def makeFile(output_path, dataFrame, fileName, label_input):
    if os.path.exists(output_path) == True:
        pass
    else:
        os.mkdir(output_path)
    if label_input == 0:
        label_input = "NG"
    elif label_input == 1:
        label_input = "OK"
    elif label_input == 2:
        label_input = "NTC"
    elif label_input == 'a':
        label_input = "ALL"
    
    dataFrame.to_csv(path_or_buf=output_path + '\\' + fileName.split('.')[0] + f"_{label_input}_." + fileName.split('.')[1], encoding='cp949')

def mv_describe(dataFrame, num, label_input):
    if label_input == '0' or label_input == '1' or label_input == '2':
        label_input = int(label_input)
        dataFrame = dataFrame.loc[dataFrame['품질상태'] == label_input]
        print(dataFrame['품질상태'].head())
        dataFrame = dataFrame.iloc[:, num : dataFrame.shape[1] : 5]
    elif label_input == 'a':
        dataFrame = dataFrame.iloc[:, num : dataFrame.shape[1] : 5]
    print(dataFrame.head())

    return dataFrame

if __name__=="__main__":
    plt.rcParams['font.family'] ='Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] =False
    #출처: https://giveme-happyending.tistory.com/168 [소연의_개발일지:티스토리]
    
    #OK, NG, NTC만 볼지, 전부 볼지 입력
    label_input = input("표시하고 싶은 라벨을 적으세요.(0, 1, 2, a): ")
    
    dataPath = os.getcwd() + "\\MLP\\datas\\data_mv_sv_dv_ut_lt_hd_with_NTC.csv"
    outputPath = os.getcwd() + "\\MLP\\EDA\\datas\\"

    dataFrame = pd.read_csv(dataPath, encoding='cp949')
    print(dataFrame.shape)
    print(dataFrame.head())

    dataFrame2 = mv_describe(dataFrame=dataFrame, num=1, label_input=label_input)
    dataFrame3 = mv_describe(dataFrame=dataFrame, num=2, label_input='a')
    
    #측정값을 describe한 데이터프레임 csv 파일 생성
    makeFileInput = input("csv파일을 생성하시겠습니까?(y, n): ")
    if makeFileInput == 'y':
        mvDescribe = dataFrame2.describe()
        makeFile(outputPath, mvDescribe, "측정값_describe_with_NTC.csv", label_input)

    shape = dataFrame2.shape[1]
    print(shape)
    lambda_func = lambda shape: 1 if shape % 8 != 0 else 0
    addition_count = lambda_func(shape)
    repeat_count = int(shape / 8) + addition_count
    print(repeat_count)
    
    count = 1
    for i in range(0, shape, repeat_count):
        fig, axes = plt.subplots(2, 4, figsize=(15,8))
        for j in range(0, 2):
            for k in range(0, 4):
                if i+(j*4)+k < shape:
                    axes[j, k].set_title(f"{dataFrame2.columns[i+(j*4)+k]}")
                    axes[j, k].hist(dataFrame2.iloc[:, i+(j*4)+k], alpha=0.75, bins=30)
                    # if(i + j < shape -1):
                    #     axes[j].axvline(x=dataFrame3.iloc[0, i+j], color='r', label='a')
        fig.canvas.manager.set_window_title(f"측정값 히스토그램 {count}")
        plt.show()
        count += 1

    print(dataFrame2[dataFrame2.iloc[:, 5] == 0])
    