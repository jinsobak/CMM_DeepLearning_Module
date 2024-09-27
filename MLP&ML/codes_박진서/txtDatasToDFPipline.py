import os
import sys
import pandas as pd
import joblib

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from preProcess import txtToDFPiplineOne
from preProcess2 import extract_data_from_file
from preprocess_judgement import DFtoModifiedDF
from EDA import PCA_visualization as pca

if __name__ == "__main__":
    #txt 파일들의 경로
    dataPath = os.getcwd() + "\\txt_datas_hd"
    #txt파일 경로에서 하나의 텍스트 파일 추출
    txtFileList = os.listdir(dataPath)
    
    labels = ['----|', '---|', '--|', '-|', '>|<', '|+', '|++', '|+++', '|++++']
    
    #폴더에서 txt파일들을 추출해서 전처리후 한 데이터프레임에 몰아서 저장
    dataFrame = pd.DataFrame()
    for index, item in enumerate(txtFileList):
        dataFrame1 = extract_data_from_file(file_path = dataPath + "\\" + item, fileName=item)
        #품번검사 후 품번이 맞지 않을 시 넘기기
        if dataFrame1['품번'][0] != '45926-4G100':
            print(f"필요품번: 45926-4G100")
            print(f"품번: {dataFrame1['품번'][0]}")
            print("품번이 맞지 않습니다.")
            continue
        dataFrame2 = DFtoModifiedDF(dataFrame = dataFrame1, fileName=item, labels=labels)
        dataFrame2.reset_index(inplace=True)
        if dataFrame2.loc[:,'품질상태'].iloc[0] != 2:
            dataFrame = pd.concat([dataFrame, dataFrame2], ignore_index=False)
    
    #데이터프레임 내용 출력 및 테스트용 csv파일로 저장
    print(dataFrame.head)
    output_path = os.getcwd() + "\\MLP&ML\\datas"
    if os.path.exists(output_path) != True:
        os.mkdir(output_path)
    dataFrame.to_csv(path_or_buf=output_path + '\\' + "txtDatasToDFTest.csv", encoding='cp949')