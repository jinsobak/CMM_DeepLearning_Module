import pandas as pd
import csv
import os

if __name__ == '__main__':
    dataset_path = os.getcwd() + "\\output_test_ld"
    data_list = os.listdir(dataset_path)
    
    #print(data_list)
    
    data = pd.read_csv(dataset_path + "\\" + data_list[0], encoding='cp949')
    
    datas = pd.DataFrame(data)
    deviation = datas['편차']
    
    print(deviation)
    