import pandas as pd
import matplotlib as plt
import os
import fill_null_values as fn

def devch(datas):
    if datas['편차'] == '-' and datas['판정'] != '-':
        datas['편차'] = abs(float(datas['측정값'])) - abs(float(datas['기준값']))
    return datas['편차']

def fill_upper_tolerance(datas):
    if datas['상한공차'] == '-':
        datas['상한공차'] = datas['기준값']
    return datas['상한공차']

def fill_lower_tolerance(datas):
    if datas['하한공차'] == '-':
        datas['하한공차'] = 0
    return datas['하한공차']

def fill_or_drop_devation(datas):
    datas.drop(datas[datas['항목'] == 'SMmf'].index, inplace=True)

    datas = datas[['품명', '도형', '항목', '측정값', '기준값', '상한공차', '하한공차','편차', '판정', '품질상태']]
    datas['편차'] = datas.apply(devch, axis = 1)
    datas.drop(datas[datas['판정']=='-'].index, inplace=True)

    return datas

def fill_ut_lt(datas):
    datas['상한공차'] = datas.apply(fill_upper_tolerance, axis=1)
    datas['하한공차'] = datas.apply(fill_lower_tolerance, axis=1)

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
        shape3 = "상한공차_" + row['도형'] + "_" + row['항목']
        shape4 = "하한공차_" + row['도형'] + "_" + row['항목']
        shape5 = "편차_" + row['도형'] + "_" + row['항목']
        new_data = pd.concat([new_data, pd.DataFrame({shape1 : [row['측정값']]})], axis=1)
        new_data = pd.concat([new_data, pd.DataFrame({shape2 : [row['기준값']]})], axis=1)
        new_data = pd.concat([new_data, pd.DataFrame({shape3 : [row['상한공차']]})], axis=1)
        new_data = pd.concat([new_data, pd.DataFrame({shape4 : [row['하한공차']]})], axis=1)
        new_data = pd.concat([new_data, pd.DataFrame({shape5 : [row['편차']]})], axis=1)
        
    new_data.drop(columns=['a'], inplace=True)

    new_data = new_data.astype(dtype='float64')

    new_data = new_data.assign(품질상태=quality)
    new_data.insert(loc=0, column='품명', value=name)

    return new_data

if __name__=="__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('mode.chained_assignment',  None)

    #dataset_path = os.getcwd() + "\\output_test_ld\\45926-4G100"
    #dataset_path = os.getcwd() + "\\output_test_sd\\45926-4G100"
    dataset_path = os.getcwd() + "\\dataset_csv\\45926-4G100"
    datalist = os.listdir(dataset_path)

    dataFrame = pd.DataFrame()
    for file in datalist:
        data = pd.read_csv(dataset_path + "\\" + file, encoding='cp949')
        datas = pd.DataFrame(data)
        datas = fill_or_drop_devation(datas)
        datas = fill_ut_lt(datas)
        datas = change_data_form(datas)
        dataFrame = pd.concat([dataFrame, datas], ignore_index=False)

    for col in dataFrame.columns:
        nanRatio = dataFrame[col].isnull().sum() / dataFrame[col].shape[0]
        #print(col + " " + str(nanRatio))
        if(nanRatio >= 0.5):
            dataFrame.drop(columns=[col], inplace=True)

    dataFrame = fn.fill_null_value_dv_mv_sv_ut_lt(dataFrame)
    print(dataFrame.isnull().sum())

    print(dataFrame['품질상태'].value_counts())

    dataFrame = dataFrame.set_index('품명')
    dataFrame.index_name = '품명'
    output_path = os.getcwd() + "\\MLP\\datas"
    if os.path.exists(output_path) == True:
        pass
    else:
        os.mkdir(output_path)
    
    dataFrame.to_csv(path_or_buf=output_path + '\\' + "data_mv_sv_dv_ut_lt_hd.csv", encoding='cp949')