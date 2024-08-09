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

    qualities = dataFrame['품질상태']
    dataFrame2 = dataFrame.iloc[:, 1:]

    shape = dataFrame2.shape[1]
    print(shape)
    subplot_count = 4
    lambda_func = lambda shape: 1 if shape % subplot_count != 0 else 0
    addition_count = lambda_func(shape)
    repeat_count = int(shape / subplot_count) + addition_count
    print(repeat_count)

    count = 1
    axes_x_size = 2
    axes_y_size = 2
    for i in range(0, shape, subplot_count):
        fig, axes = plt.subplots(axes_x_size, axes_y_size, figsize=(15,8))
        for j in range(0, axes_x_size):
            for k in range(0, axes_y_size):
                if i+(j*axes_y_size)+k < shape:
                    axes[j, k].set_title(f"{dataFrame2.columns[i+(j*axes_y_size)+k]}")
                    axes[j, k].scatter(x=dataFrame2.iloc[:, i+(j*axes_y_size)+k], y=qualities, alpha=0.3)
        fig.canvas.manager.set_window_title(f"편차 산점도 {count}")
        plt.show()
        count += 1