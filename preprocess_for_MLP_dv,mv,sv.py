import pandas as pd  # Pandas를 불러옵니다.
import matplotlib as plt  # Matplotlib를 불러옵니다. 하지만 사용되지 않습니다.
import os  # 운영체제와 상호작용하기 위해 os 모듈을 불러옵니다.
import csv  # CSV 파일을 다루기 위해 csv 모듈을 불러옵니다.

# 편차 계산 함수입니다.
def devch(datas):
    # '편차'가 '-'이고 '판정'이 '-'이 아닌 경우에만 편차를 계산합니다.
    if datas['편차'] == '-' and datas['판정'] != '-':
        datas['편차'] = float(datas['기준값']) - float(datas['측정값'])
    return datas['편차']

# 데이터프레임에서 특정 조건을 만족하는 행을 제거하고 다시 정렬하는 함수입니다.
def fill_or_drop_devation(datas):
    # '항목'이 'SMmf'인 행을 제거합니다.
    datas.drop(datas[datas['항목'] == 'SMmf'].index, inplace=True)

    # 필요한 열만 선택합니다.
    datas = datas[['품명', '도형', '항목', '측정값', '기준값','편차', '판정', '품질상태']]

    # '편차' 열에 대해 편차를 계산합니다.
    datas['편차'] = datas.apply(devch, axis = 1)

    # '판정'이 '-'인 행을 제거합니다.
    datas.drop(datas[datas['판정']=='-'].index, inplace=True)

    return datas

# 데이터 형식을 변경하는 함수입니다.
def change_data_form(datas):
    # '품질상태'가 'OK'인 경우 1로, 아닌 경우 0으로 변경합니다.
    quality = datas["품질상태"][0]
    if quality == "OK":
        quality = 1
    else:
        quality = 0
    
    # '품명'을 선택합니다.
    name = datas["품명"][0]

    # 새로운 데이터프레임을 생성합니다.
    new_data = pd.DataFrame({'a' : [0]})

    # 각 행에 대해 반복하며 측정값, 기준값, 편차의 열을 만들어 추가합니다.
    for index, row in datas.iterrows():
        shape1 = "측정값_" + row['도형'] + "_" + row['항목']
        shape2 = "기준값_" + row['도형'] + "_" + row['항목']
        shape3 = "편차_" + row['도형'] + "_" + row['항목']
        new_data = pd.concat([new_data, pd.DataFrame({shape1 : [row['측정값']]})], axis=1)
        new_data = pd.concat([new_data, pd.DataFrame({shape2 : [row['기준값']]})], axis=1)
        new_data = pd.concat([new_data, pd.DataFrame({shape3 : [row['편차']]})], axis=1)
    new_data.drop(columns=['a'], inplace=True)

    # '품질상태' 열을 추가하고 값을 설정합니다.
    new_data = new_data.assign(품질상태=quality)

    # '품명' 열을 추가하고 인덱스로 설정합니다.
    new_data.insert(loc=0, column='품명', value=name)
    new_data = new_data.set_index('품명')
    new_data.index_name = '품명'

    return new_data

if __name__=="__main__":
    # 출력되는 열과 행의 수를 제한합니다.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('mode.chained_assignment',  None)

    # 데이터셋의 경로를 설정합니다.
    #datasetPath = os.getcwd() + "\\output_test_ld\\45926-4G100"
    datasetPath = os.getcwd() + "\\references\\천세진 교수님 멘토링 자료\\datasets\\large_data"
    
    datalist = os.listdir(datasetPath)

    # 빈 데이터프레임을 생성합니다.
    dataFrame = pd.DataFrame()

    # 각 파일에 대해 반복하여 데이터프레임에 추가합니다.
    for file in datalist:
        data = pd.read_csv(datasetPath + "\\" + file, encoding='cp949')
        datas = pd.DataFrame(data)
        datas = fill_or_drop_devation(datas)
        datas = change_data_form(datas)
        dataFrame = pd.concat([dataFrame, datas], ignore_index=False)

    # 출력 경로를 설정합니다.
    output_path = os.getcwd() + "\\MLP\\test"

    # 출력 경로가 존재하지 않으면 새로 생성합니다.
    if os.path.exists(output_path) == True:
        pass
    else:
        os.mkdir(output_path)
    
    # 데이터를 CSV 파일로 저장합니다.
    dataFrame.to_csv(path_or_buf=output_path + '\\' + "data_mv,sv,dv_ld.csv", encoding='cp949')
