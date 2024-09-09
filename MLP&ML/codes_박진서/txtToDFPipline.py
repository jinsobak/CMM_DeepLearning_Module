import os
import sys
import pandas as pd

#sys.path.append(os.getcwd() + "/MLP&ML/EDA")
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from preProcess import txtToDFPiplineOne
from preProcess2 import extract_data_from_file
from preprocess_judgement import DFtoModifiedDF
from EDA import PCA_visualization as pca

if __name__ == "__main__":
    dataPath = os.getcwd() + "\\txt_datas_hd"
    testDataPath = os.getcwd() + "\\MLP&ML\\datas\\data_jd_hd_no_NTC.csv"
    
    txtFileList = os.listdir(dataPath)
    txtFileName = txtFileList[3]
    
    dataFrame1 = extract_data_from_file(file_path = dataPath + "\\" + txtFileName, fileName=txtFileName)
    print(txtFileName)
    
    labels = ['----|', '---|', '--|', '-|', '>|<', '|+', '|++', '|+++', '|++++']
    dataFrame2 = DFtoModifiedDF(dataFrame=dataFrame1, fileName=txtFileName, labels=labels)  
    dataFrame2.reset_index(inplace=True)
    print(dataFrame2.head)
    
    output_path = os.getcwd() + "\\MLP&ML\\datas"
    if os.path.exists(output_path) != True:
        os.mkdir(output_path)
    dataFrame2.to_csv(path_or_buf=output_path + '\\' + "txtToDFTest.csv", encoding='cp949')
    
    pcaBaseDataFrame = pd.read_csv(testDataPath, encoding='cp949')
    
    # num_pca_components = 4
    
    # pca_df_target, pca_df_fileName, pca_df_datas = pca.distributeDataFrame(dataFrame=dataFrame2)
    # pca_df_base_target, pca_df_base_fileName, pca_df_base_datas = pca.distributeDataFrame(dataFrame=pcaBaseDataFrame)
    
    # pca_df_base_scaled = pca.performStandScalar(df_datas=pca_df_base_datas)
    # pca_df_scaled = pca.performStandScalar(df_datas=pca_df_datas)
    
    # pca_dataFrame = pca.make_pca_dataFrame(data_scaled_base=pca_df_base_scaled, data_scaled=pca_df_scaled, 
    #                                        data_target=pca_df_target, data_fileName= pca_df_fileName, 
    #                                        num_components=num_pca_components)
    
    
    