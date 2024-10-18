import os
import sys
import pandas as pd
import joblib
import streamlit as st
import tempfile

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from preProcess2 import extract_data_from_file
from preprocess_judgement import DFtoModifiedDF
import PCA_visualization as pca

def CheckFileNum(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name  # 임시 파일 경로
        
    dataFrame1 = extract_data_from_file(file_path = temp_file_path, fileName=file.name)
    #품번검사 후 품번이 맞지 않을 시 넘기기
    if dataFrame1['품번'][0] != '45926-4G100':
        st.write(f"파일 이름: {file.name} 품번: {dataFrame1['품번'][0]}")
        # print(f"파일 이름: {fileName}")
        # print(f"필요품번: 45926-4G100")
        # print(f"품번: {dataFrame1['품번'][0]}")
        # print("품번이 맞지 않습니다.")
        return None
    return dataFrame1
    
def ModifyEarlyPreprocessedDF(dataFrame, fileName):
    labels = ['----|', '---|', '--|', '-|', '>|<', '|+', '|++', '|+++', '|++++']

    dataFrame_modified = DFtoModifiedDF(dataFrame = dataFrame, fileName=fileName, labels=labels)
    dataFrame_modified.reset_index(inplace=True)
    
    return dataFrame_modified

def makePreprocessedDf(txtFileList):
    dataFrame = pd.DataFrame()
    
    for index, item in enumerate(txtFileList):
        #st.write(item)
        dataFrame1 = CheckFileNum(file=item)
        if dataFrame1 is not None:
            dataFrame2 = ModifyEarlyPreprocessedDF(dataFrame=dataFrame1, fileName=item.name)
            if dataFrame2.loc[:,'품질상태'].iloc[0] != 2:
                dataFrame = pd.concat([dataFrame, dataFrame2], ignore_index=False)
    dataFrameNaFilled = dataFrame.fillna(value=0)
    dataFrameResetIndex = dataFrameNaFilled.reset_index()
    dataFrameResetIndex.drop(columns=['index'], inplace=True)
    
    return dataFrameResetIndex
    
if __name__ == "__main__":
    #txt 파일들의 경로
    dataPath = os.getcwd() + "\\txt_datas_hd"
    #txt파일 경로에서 하나의 텍스트 파일 추출
    txtFileList = os.listdir(dataPath)
    
    #모델 제작여부 묻기
    be_scalar_model_save = input("StandardScalar 모델을 저장하시겠습니까? ")
    be_pcaModel_save = input("pca 모델을 저장하시겠습니까? ")
    
    #폴더에서 txt파일들을 추출해서 전처리후 한 데이터프레임에 몰아서 저장
    dataFrame = makePreprocessedDf(txtFileList=txtFileList)

    pca_df_target, pca_df_fileName, pca_df_datas = pca.distributeDataFrame(dataFrame=dataFrame)
    
    #데이터프레임 내용 출력 및 테스트용 csv파일로 저장
    print(dataFrame.head)
    output_path = os.getcwd() + "\\MLP&ML\\datas"
    if os.path.exists(output_path) != True:
        os.mkdir(output_path)
    pca_df_datas.to_csv(path_or_buf=output_path + '\\' + "pca_datas_test.csv", encoding='cp949')
    
    pca_scalar_model = pca.Make_StandScalar_model(df_datas=pca_df_datas)
    df_scaled = pca_scalar_model.transform(pca_df_datas)
    df_scaled = pd.DataFrame(df_scaled, columns=pca_df_datas.columns)
    
    if be_scalar_model_save == 'y':
        scalar_model_save_path = os.getcwd() + "\\MLP&ML\\Skl_models\\Scalar"
        scalar_model_name = "scalar_model"
        pca.save_model(pca_scalar_model, scalar_model_save_path, scalar_model_name)
    
    pca_num_components = 7
    
    pca_model = pca.Make_pca_model(data_scaled = df_scaled, num_components = pca_num_components)
    
    if be_pcaModel_save == 'y':
        pca_model_save_path = os.getcwd() + "\\MLP&ML\\Skl_models\\Pca"
        pca_model_name = f"pca_model_{pca_num_components}" 
        pca.save_model(pca_model, pca_model_save_path, pca_model_name)
    
    df_pca = pca.make_pca_dataFrame(data_scaled=df_scaled, data_target=pca_df_target, data_fileName=pca_df_fileName, num_components=pca_num_components, pca_model= pca_model)
    