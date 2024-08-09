import os
import csv
import pandas as pd
import sys
import re
from preProcess2 import extract_data_from_file

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