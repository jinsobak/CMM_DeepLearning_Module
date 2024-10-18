import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

def distributeDataFrame(dataFrame):
    df_target = dataFrame.loc[:,"품질상태"]
    df_fileName = dataFrame.loc[:,'파일명']
    df_datas = dataFrame.drop(columns=["파일명", "품질상태"])
    
    return df_target, df_fileName, df_datas

def Make_StandScalar_model(df_datas):
    scalar = StandardScaler()
    scalar.fit(df_datas)
    
    return scalar

def Make_pca_model(data_scaled, num_components):
    pca_model = PCA(n_components=num_components)
    pca_model.fit(data_scaled)
    
    return pca_model

def make_pca_dataFrame(data_scaled, data_target, data_fileName, num_components, pca_model = None):
    #PCA진행
    pca = pca_model.transform(data_scaled)
    df_pca_column_names = [f'component_{i}' for i in range(0, num_components)]
    df_pca = pd.DataFrame(pca, columns = df_pca_column_names)
    df_pca['품질상태'] = data_target
    df_pca['파일명'] = data_fileName
    df_pca= df_pca.set_index('파일명')
    df_pca.index.name = '파일명'
    
    print(pca_model.explained_variance_ratio_)
    print(pca_model.explained_variance_ratio_.sum())
    print(df_pca.head())
    
    return df_pca

def save_model(model, model_save_path, model_name):
    if os.path.exists(model_save_path) != True:
        os.mkdir(model_save_path)
    joblib.dump(model, model_save_path + "\\" + model_name + ".pkl")

if __name__=="__main__":
    plt.rcParams['font.family'] ='Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] =False
    #출처: https://giveme-happyending.tistory.com/168 [소연의_개발일지:티스토리]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    be_visualize_2d = input("시각화 여부를 선택하세요(y, n) ")
    be_makeFile = input("파일 생성 여부를 선택하세요(y, n) ")
    be_scalar_model_save = input("StandardScalar 모델을 저장하시겠습니까? ")
    be_pcaModel_save = input("pca 모델을 저장하시겠습니까? ")
    
    dataPath = os.getcwd() + "\\MLP&ML\\datas\\"
    outputPath = os.getcwd() + "\\MLP&ML\\datas\\"

    #dataFileName = 'data_jd_hd_no_NTC'
    #해당 데이터 기준 컴포넌트 4개: 28%, 컴포넌트 26개: 95%
    dataFileName = 'data_jd_hd_delete_material_no_NTC'
    #해당 데이터 기준 컴포넌트 4개: 43%, 컴포넌트 7개: 61%, 컴포넌트 17개: pca_3
    #dataFileName = 'data_mv_sv_dv_ut_lt_hd_no_NTC'
    #해당 데이터 기준 컴포넌트 4개: 49%, 컴포넌트 26개: 95%
    
    dataFrame = pd.read_csv(dataPath + dataFileName + '.csv', encoding='cp949')
    print(dataFrame.shape)
    #print(dataFrame.index)
    
    #데이터 분할
    df_target, df_fileName, df_datas = distributeDataFrame(dataFrame=dataFrame)
    
    #정규화
    scalar_model = Make_StandScalar_model(df_datas=df_datas)
    df_scaled = scalar_model.transform(df_datas)
    df_scaled = pd.DataFrame(df_scaled, columns=df_datas.columns)
    
    num_components = 7
    pca_model = Make_pca_model(data_scaled = df_scaled, num_components = num_components)
    df_pca = make_pca_dataFrame(data_scaled=df_scaled, data_target=df_target, data_fileName=df_fileName, num_components=num_components, pca_model= pca_model)
    
    if be_visualize_2d == 'y':
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
    
    if be_scalar_model_save == 'y':
        scalar_model_save_path = os.getcwd() + "\\MLP&ML\\Skl_models\\Scalar"
        scalar_model_name = "scalar_model"
        save_model(scalar_model, scalar_model_save_path, scalar_model_name)
    
    if be_pcaModel_save == 'y':
        pca_model_save_path = os.getcwd() + "\\MLP&ML\\Skl_models\\Pca"
        pca_model_name = f"pca_model" 
        save_model(pca_model, pca_model_save_path, pca_model_name)
    
    if be_makeFile == 'y':
        dataName = f"{dataFileName}_pca_component_{num_components}.csv"
        print(dataName)
        df_pca.to_csv(path_or_buf=outputPath + '\\' + dataName, encoding='cp949')