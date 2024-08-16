import pandas as pd
import matplotlib.pyplot as plt
import os
import describe_mv as des
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def make_pca_dataFrame(data_scaled, num_components):
    #PCA진행
    num_components = num_components
    pca_model = PCA(n_components=num_components)
    pca_model.fit(data_scaled)
    
    pca = pca_model.transform(data_scaled)
    df_pca_column_names = [f'component_{i}' for i in range(0, num_components)]
    df_pca = pd.DataFrame(pca, columns=  df_pca_column_names)
    df_pca['품질상태'] = df_target
    df_pca['파일명'] = df_fileName
    df_pca= df_pca.set_index('파일명')
    df_pca.index.name = '파일명'
    
    print(pca_model.explained_variance_ratio_)
    print(pca_model.explained_variance_ratio_.sum())
    print(df_pca.head())
    
    return df_pca

if __name__=="__main__":
    plt.rcParams['font.family'] ='Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] =False
    #출처: https://giveme-happyending.tistory.com/168 [소연의_개발일지:티스토리]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    be_visualize_2d = input("시각화 여부를 선택하세요(y, n)")
    be_makeFile = input("파일 생성 여부를 선택하세요(y, n)")
    
    dataPath = os.getcwd() + "\\MLP&ML\\datas\\"
    outputPath = os.getcwd() + "\\MLP&ML\\datas\\"

    #dataFileName = 'data_jd_hd_no_NTC'
    #해당 데이터 기준 컴포넌트 4개: 28%, 컴포넌트 26개: 95%
    #dataFileName = 'data_jd_hd_delete_material_no_NTC'
    #해당 데이터 기준 컴포넌트 4개: 43%, 컴포넌트 7개: 61%, 컴포넌트 17개: pca_3
    dataFileName = 'data_mv_sv_dv_ut_lt_hd_no_NTC'
    #해당 데이터 기준 컴포넌트 4개: 49%, 컴포넌트 26개: 95%
    
    dataFrame = pd.read_csv(dataPath + dataFileName + '.csv', encoding='cp949')
    print(dataFrame.shape)
    
    df_target = dataFrame.loc[:,"품질상태"]
    df_fileName = dataFrame.loc[:, '파일명']
    df_datas = dataFrame.drop(columns=["파일명", "품질상태"])
    
    #정규화
    scalar = StandardScaler()
    scalar.fit(df_datas)
    df_scaled = scalar.transform(df_datas)
    df_scaled = pd.DataFrame(df_scaled, columns=df_datas.columns)
    
    num_components = 26
    df_pca = make_pca_dataFrame(data_scaled=df_scaled, num_components=num_components)
    
    if be_visualize_2d == 'y':
        
        # #target 별 분리
        # df_pca_0 = df_pca[df_pca['품질상태'] == 0]
        # df_pca_1 = df_pca[df_pca['품질상태'] == 1]

        # #target 별 시각화
        # plt.figure(figsize=[10, 6])
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
        tsne_df['파일명'] = df_fileName
        tsne_df = tsne_df.set_index('파일명')
        tsne_df.index.name = '파일명'
        
        # target 별 분리
        tsne_df_0 = tsne_df[tsne_df['품질상태'] == 0]
        tsne_df_1 = tsne_df[tsne_df['품질상태'] == 1]

        # target 별 시각화
        plt.figure(figsize=[10, 6])
        plt.scatter(tsne_df_0['component_0'], tsne_df_0['component_1'], color = 'blue', alpha=0.7, label = 0)
        plt.scatter(tsne_df_1['component_0'], tsne_df_1['component_1'], color = 'yellow', alpha=0.7, label = 1)
        plt.xlabel('component_0')
        plt.ylabel('component_1')
        plt.legend()
        plt.show()
        
    if be_makeFile == 'y':
        #dataName = 'data_mv_sv_dv_ut_lt_pca_no_NTC_2.csv'
        dataName = f"{dataFileName}_pca_component_{num_components}.csv"
        print(dataName)
        df_pca.to_csv(path_or_buf=outputPath + '\\' + dataName, encoding='cp949')