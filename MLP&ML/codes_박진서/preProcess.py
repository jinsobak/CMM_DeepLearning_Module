import os
import pandas as pd
from preProcess2 import extract_data_from_file

def txtToDFPiplineOne(dataPath):
    txtFileList = os.listdir(dataPath)

    dataframe = extract_data_from_file(file_path = dataPath + "\\" + txtFileList[0], fileName=txtFileList[0])
        
    return dataframe, txtFileList[0][0:-4] + ".txt"

if __name__ == "__main__":
    dataset_path = os.getcwd() + "\\txt_datas_hd"
    output_path = os.getcwd() + "\\csv_datas_hd"
    data_list = os.listdir(dataset_path)

    print(os.path.exists(output_path))
    if os.path.exists(output_path) == True:
        pass
    else:
        os.mkdir(output_path)

    for item in data_list:
        dataFrame = extract_data_from_file(file_path=dataset_path + "\\" + item, fileName=item)
        output_path2 = output_path + '\\' + str(dataFrame['품번'][0]) + '\\'
        if os.path.exists(output_path2) == True:
            pass
        else:
            os.mkdir(output_path2)
        #dataFrame.to_csv(path_or_buf=output_path2 + item[:-4]+".csv", encoding="cp949")
        print("done")