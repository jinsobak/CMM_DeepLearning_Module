import pandas as pd
import matplotlib.pyplot as plt
import os
import describe_mv as des
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if __name__=="__main__":
    plt.rcParams['font.family'] ='Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] =False
    #출처: https://giveme-happyending.tistory.com/168 [소연의_개발일지:티스토리]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    dataPath = os.getcwd() + "\\MLP\\datas\\data_jd_no_NTC.csv"
    #dataPath = os.getcwd() + "\\MLP\\datas\\data_jd_hd2_delete_material_no_NTC.csv"
    #dataPath = os.getcwd() + "\\MLP\\datas\\data_mv_sv_dv_ut_lt_hd_no_NTC.csv"
    outputPath = os.getcwd() + "\\MLP\\EDA\\datas\\"

    dataFrame = pd.read_csv(dataPath, encoding='cp949')
    print(dataFrame.shape)
    
    df_target = dataFrame.loc[:,"품질상태"]
    df_datas = dataFrame.drop(columns=["파일명", "품질상태"])
    
    #정규화
    scalar = StandardScaler()
    scalar.fit(df_datas)
    df_scaled = scalar.transform(df_datas)
    df_scaled = pd.DataFrame(df_scaled, columns=df_datas.columns)
    
    #PCA진행
    pca_model = PCA(n_components=4)
    pca_model.fit(df_scaled)
    
    pca = pca_model.transform(df_scaled)
    df_pca = pd.DataFrame(pca, columns=['component_0', 'component_1', 'component_2', 'component_3'])
    df_pca['품질상태'] = df_target
    
    print(pca_model.explained_variance_ratio_)
    print(df_pca.head())
    
    # #target 별 분리
    # df_pca_0 = df_pca[df_pca['품질상태'] == 0]
    # df_pca_1 = df_pca[df_pca['품질상태'] == 1]

    # #target 별 시각화
    # plt.scatter(df_pca_0['component_0'], df_pca_0['component_1'], color = 'blue', alpha=1, label = 0)
    # plt.scatter(df_pca_1['component_0'], df_pca_1['component_1'], color = 'yellow', alpha=1, label = 1)
    # plt.xlabel('component_0')
    # plt.ylabel('component_1')
    # plt.legend()
    # plt.show()
    
    tsne = TSNE(n_components=2).fit_transform(df_pca)
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