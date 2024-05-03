import pandas as pd
import os
import csv


def devch(datas):                                                                               #데이터의 '편차'를 계산하여 업데이트하는 역할
    if datas['편차'] == '-' and datas['판정'] != '-':                                           # '편차'가 '-'이고, '판정'이 '-'가 아닌 경우, '편차'를 '기준값'과 '측정값'의 차이로 계산하여 업데이트
        datas['편차'] = float(datas['기준값']) - float(datas['측정값'])
    return datas['편차']

def fill_or_drop_devation(datas):                                                              # 데이터에서 필요 없는 행을 제거하고, '편차'를 업데이트하는 함수
    datas.drop(datas[datas['항목'] == 'SMmf'].index, inplace=True)                              # '항목'이 'SMmf'인 행 제거

    datas = datas[['품명', '도형', '항목', '측정값', '기준값', '편차', '판정', '품질상태']]        # 데이터프레임을 특정 컬럼으로 필터링
    datas['편차'] = datas.apply(devch, axis=1)                                                  # devch 함수를 사용하여 '편차' 컬럼 업데이트
    datas.drop(datas[datas['판정'] == '-'].index, inplace=True)                                 # '판정'이 '-'인 행을 제거

    return datas

def change_data_form(datas):                                                                    # 데이터의 형식을 머신러닝 모델 학습에 적합하게 변환하는 함수
    quality = datas["품질상태"][0]                                                               # '품질상태' 컬럼의 첫 번째 값에 따라 1 또는 0으로 변환
    quality = 1 if quality == "OK" else 0   

    name = datas["품명"][0]

    new_data = pd.DataFrame({'a': [0]})                                                         # 새로운 데이터프레임을 생성
    for index, row in datas.iterrows():
        shape1 = "측정값_" + row['도형'] + "_" + row['항목']
        shape2 = "기준값_" + row['도형'] + "_" + row['항목']
        shape3 = "편차_" + row['도형'] + "_" + row['항목']
        new_data = pd.concat([new_data, pd.DataFrame({shape1: [row['측정값']]})], axis=1)       # 각 '도형'과 '항목'에 따른 '측정값', '기준값', '편차'를 새로운 컬럼으로 추가
        new_data = pd.concat([new_data, pd.DataFrame({shape2: [row['기준값']]})], axis=1)
        new_data = pd.concat([new_data, pd.DataFrame({shape3: [row['편차']]})], axis=1)
    new_data.drop(columns=['a'], inplace=True)                                                  #불필요한 초기 컬럼('a')을 삭제

    new_data = new_data.assign(품질상태=quality)                                                # '품질상태' 컬럼을 추가
    new_data.insert(loc=0, column='품명', value=name)                                           # '품명' 컬럼을 추가하고, 인덱스로 설정
    new_data = new_data.set_index('품명')
    new_data.index_name = '품명'

    return new_data

if __name__=="__main__":
    pd.set_option('display.max_columns', None)                                                 # pandas 라이브러리 설정을 변경하여 DataFrame 출력 시 최대 컬럼 수와 최대 로우 수를 제한하지 않도록 설정
    pd.set_option('display.max_rows', None)                                                    # DataFrame을 출력할 때 모든 컬럼과 로우가 표시 
    pd.set_option('mode.chained_assignment',  None)                                            # 경고 메시지를 무시하기 위한 설정.'SettingWithCopyWarning' 경고를 방지하기 위해 사용

    datasetPath = os.getcwd() + "\\output_test_ld\\45926-4G100"                                # 현재 작업 디렉토리의 경로를 가져온 후, 그 경로에 있는 "output_test_ld\\45926-4G100" 폴더의 경로를 생성
    datalist = os.listdir(datasetPath)                                                         # datasetPath 경로에 있는 모든 파일의 이름을 리스트로 가져옴

    dataFrame = pd.DataFrame()                                                                 # 빈 DataFrame을 생성. 이 DataFrame은 나중에 각 파일에서 읽은 데이터를 저장하는 데 사용
    
    for file in datalist:                                                                      # datalist에 있는 각 파일에 대해 반복
        data = pd.read_csv(datasetPath + "\\" + file, encoding='cp949')                        # 각 파일을 읽어들여 DataFrame으로 변환. 파일 인코딩으로 'cp949'를 사용
        datas = pd.DataFrame(data)
        datas = fill_or_drop_devation(datas)                                                   # fill_or_drop_devation 함수를 호출하여 '편차' 관련 처리 진행.
        datas = change_data_form(datas)                                                        # change_data_form 함수를 호출하여 데이터 형태를 변환,데이터를 머신러닝 모델 학습에 적합한 형태로 변환
        dataFrame = pd.concat([dataFrame, datas], ignore_index=False)                          # 변환된 데이터를 dataFrame에 추가. ignore_index=False로 설정하여 인덱스를 유지

    output_path = os.getcwd() + "\\MLP\\test"                                                  # 데이터를 저장할 경로를 설정. 현재 작업 디렉토리 아래에 "MLP\\test" 폴더의 경로를 생성.
    if os.path.exists(output_path) == True:                                                    # 해당 경로가 이미 존재하는지 확인. 존재하지 않는 경우 새로운 폴더를 생성
        pass
    else:
        os.mkdir(output_path)

    dataFrame.to_csv(path_or_buf=output_path + '\\' + "data_mv,sv,dv_ld.csv", encoding='cp949') # 최종적으로 처리된 데이터를 CSV 파일로 저장. 파일 경로와 이름을 설정하고, 인코딩으로 'cp949'를 사용