import os
import csv
import pandas as pd
import sys
import re
from preProcess2 import extract_data_from_file

#dataset_path = os.getcwd() + "\\datasets"
#dataset_path = os.getcwd() + "\\april_week1\\small_data"
dataset_path = os.getcwd() + "\\april_week1\\large_data"
#output_path = os.getcwd() + "\\output_test\\"
#output_path = os.getcwd() + "\\april_week1\\output_test_sd\\"
output_path = os.getcwd() + "\\april_week1\\output_test_ld\\"
data_list = os.listdir(dataset_path)
#print(data_list)

# with open(dataset_path + "\\" + data_list[0], 'r', encoding="EUC-KR") as f:
#     data = f.readlines();
# for line in data:
#     print(line)

# dataFrame = extract_data_from_file(dataset_path + "\\" + data_list[0])
#print(dataFrame)

print(os.path.exists(output_path))
if os.path.exists(output_path) == True:
    pass
else:
    os.mkdir(output_path)

for item in data_list:
    dataFrame = extract_data_from_file(dataset_path + "\\" + item)
    dataFrame.to_csv(path_or_buf=output_path + item[:-4]+".csv", encoding="cp949")
    print("done")