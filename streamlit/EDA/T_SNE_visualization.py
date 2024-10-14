import pandas as pd
import matplotlib.pyplot as plt
import os
import describe_mv as des
from sklearn.manifold import TSNE

if __name__=="__main__":
    plt.rcParams['font.family'] ='Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] =False
    #출처: https://giveme-happyending.tistory.com/168 [소연의_개발일지:티스토리]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    #dataPath = os.getcwd() + "\\MLP\\datas\\data_jd_no_NTC.csv"
    dataPath = os.getcwd() + "\\MLP\\datas\\data_jd_hd2_delete_material_no_NTC.csv"
    #dataPath = os.getcwd() + "\\MLP\\datas\\data_mv_sv_dv_ut_lt_hd_no_NTC.csv"
    outputPath = os.getcwd() + "\\MLP\\EDA\\datas\\"

    dataFrame = pd.read_csv(dataPath, encoding='cp949')
    print(dataFrame.shape)
    
    df_target = dataFrame.loc[:,"품질상태"]
    df_datas = dataFrame.drop(columns=["파일명", "품질상태"])
    
    tsne = TSNE(n_components=2, random_state=1).fit_transform(df_datas)
    print(tsne.shape)
    
    tsne_df = pd.DataFrame(tsne, columns=['component_0', 'component_1'])
    tsne_df['품질상태'] = df_target
    
    # target 별 분리
    tsne_df_0 = tsne_df[tsne_df['품질상태'] == 0]
    tsne_df_1 = tsne_df[tsne_df['품질상태'] == 1]

    # target 별 시각화
    plt.figure(figsize=[10, 6])
    plt.scatter(tsne_df_0['component_0'], tsne_df_0['component_1'], color = 'blue', alpha=1, label = 0)
    plt.scatter(tsne_df_1['component_0'], tsne_df_1['component_1'], color = 'yellow', alpha=1, label = 1)
    plt.xlabel('component_0')
    plt.ylabel('component_1')
    plt.legend()
    plt.show()