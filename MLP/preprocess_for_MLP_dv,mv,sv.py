import pandas as pd
import matplotlib as plt
import os
import csv

def devch(datas):
    if datas['편차'] == '-' and datas['판정'] != '-':
        datas['편차'] = float(datas['기준값']) - float(datas['측정값'])
    return datas['편차']

def fill_or_drop_devation(datas):
    datas.drop(datas[datas['항목'] == 'SMmf'].index, inplace=True)

    datas = datas[['품명', '도형', '항목', '측정값', '기준값','편차', '판정', '품질상태']]
    datas['편차'] = datas.apply(devch, axis = 1)
    datas.drop(datas[datas['판정']=='-'].index, inplace=True)

    return datas

def change_data_form(datas):
    quality = datas["품질상태"][0]
    if(quality == "OK"):
        quality = 1
    else:
        quality = 0
    
    name = datas["품명"][0]

    new_data = pd.DataFrame({'a' : [0]})
    for index, row in datas.iterrows():
        shape1 = "측정값_" + row['도형'] + "_" + row['항목']
        shape2 = "기준값_" + row['도형'] + "_" + row['항목']
        shape3 = "편차_" + row['도형'] + "_" + row['항목']
        new_data = pd.concat([new_data, pd.DataFrame({shape1 : [row['측정값']]})], axis=1)
        new_data = pd.concat([new_data, pd.DataFrame({shape2 : [row['기준값']]})], axis=1)
        new_data = pd.concat([new_data, pd.DataFrame({shape3 : [row['편차']]})], axis=1)
        # new_data[shape1] = row['측정값']
        # new_data[shape2] = row['기준값']
        # new_data[shape3] = row['편차']
    new_data.drop(columns=['a'], inplace=True)

    new_data = new_data.assign(품질상태=quality)
    new_data.insert(loc=0, column='품명', value=name)

    new_data = new_data.set_index('품명')
    new_data.index_name = '품명'

    return new_data

if __name__=="__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('mode.chained_assignment',  None)

    datasetPath = os.getcwd() + "\\output_test_ld\\45926-4G100"
    datalist = os.listdir(datasetPath)

    dataFrame = pd.DataFrame()
    for file in datalist:
        data = pd.read_csv(datasetPath + "\\" + file, encoding='cp949')
        datas = pd.DataFrame(data)
        datas = fill_or_drop_devation(datas)
        datas = change_data_form(datas)
        dataFrame = pd.concat([dataFrame, datas], ignore_index=False)

    output_path = os.getcwd() + "\\MLP\\test"
    if os.path.exists(output_path) == True:
        pass
    else:
        os.mkdir(output_path)
    
    dataFrame.to_csv(path_or_buf=output_path + '\\' + "data_mv,sv,dv_ld.csv", encoding='cp949')