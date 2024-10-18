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
        st.write("품번이 맞지 않습니다.")
        return None
    return dataFrame1

def ModifyEarlyPreprocessedDF(dataFrame, fileName):
    labels = ['----|', '---|', '--|', '-|', '>|<', '|+', '|++', '|+++', '|++++']

    dataFrame_modified = DFtoModifiedDF(dataFrame = dataFrame, fileName=fileName, labels=labels)
    dataFrame_modified.reset_index(inplace=True)
    
    return dataFrame_modified

def MakePreprocessedDf(txtFile):
    dataFrame1 = CheckFileNum(file=txtFile)

    if dataFrame1 is not None:
        dataFrame2 = ModifyEarlyPreprocessedDF(dataFrame=dataFrame1, fileName=txtFile.name)

    dataFrameNaFilled = dataFrame2.fillna(value=0)
    dataFrameResetIndex = dataFrameNaFilled.reset_index()
    dataFrameResetIndex.drop(columns=['index'], inplace=True)
    
    return dataFrameResetIndex

if __name__ == "__main__":
    #txt 파일들의 경로
    dataPath = os.getcwd() + "\\txt_datas_hd"
    #학습에 사용한 데이터들을 csv화 시킨 csv파일
    testDataPath = os.getcwd() + "\\MLP&ML\\datas\\data_jd_hd_no_NTC.csv"
    #txt파일 경로에서 하나의 텍스트 파일 추출
    txtFileList = os.listdir(dataPath)
    txtFileName = txtFileList[3]
    
    #1차 전처리
    dataFrame1 = extract_data_from_file(file_path = dataPath + "\\" + txtFileName, fileName=txtFileName)
    print(txtFileName)
    print(str(dataFrame1['품번'][0]))
    
    if dataFrame1['품번'][0] != '45926-4G100':
        print("품번이 맞지 않습니다.")
        exit()
    
    #2차 전처리
    labels = ['----|', '---|', '--|', '-|', '>|<', '|+', '|++', '|+++', '|++++']
    dataFrame2 = DFtoModifiedDF(dataFrame=dataFrame1, fileName=txtFileName, labels=labels)  
    dataFrame2.reset_index(inplace=True)
    print(dataFrame2.head)
    
    #pca 데이터의 컴포넌트 개수 조절
    num_pca_components = 7
    
    #pca를 적용하기위해 베이스 데이터프레임과, 예측에 사용할 데이터프레임을 품질상태, 파일이름, 데이터로 나눔
    pca_df_target, pca_df_fileName, pca_df_datas = pca.distributeDataFrame(dataFrame=dataFrame2)
    
    #저장되어 있는 StandardScalar 모델을 통해 예측용 데이터프레임에 대해 정규화
    pca_scalar_model = joblib.load(os.getcwd() +"\\MLP&ML\\Skl_models\\Scalar\\scalar_model.pkl")
    pca_df_scaled = pca_scalar_model.transform(pca_df_datas)
    pca_df_scaled = pd.DataFrame(pca_df_scaled, columns = pca_df_datas.columns)
    
    #저장되어 있는 PCA모델을 통해 정규화된 예측용 데이터프레임에 PCA기법 수행
    pca_model = joblib.load(os.getcwd() + f"\\MLP&ML\\Skl_models\\Pca\\pca_model_{num_pca_components}.pkl")
    pca_dataFrame = pca.make_pca_dataFrame(data_scaled = pca_df_scaled, data_target = pca_df_target, 
                                           data_fileName = pca_df_fileName, num_components = num_pca_components,
                                           pca_model = pca_model)
    
    
    