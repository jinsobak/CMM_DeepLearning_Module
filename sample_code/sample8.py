import pandas as pd
import numpy as np
import os

def process_file(file_path, standard_value, upper_tolerance, lower_tolerance):
    """
    파일에서 데이터를 읽어와 기준값, 측정값, 편차, 상한 허용오차, 하한 허용오차를 계산합니다.
    """
    # 파일에서 데이터 읽기, 여기에서 encoding을 변경합니다.
    # 'ISO-8859-1', 'cp1252', 또는 파일에 맞는 다른 인코딩을 시도해보세요.
    measurement_values = pd.read_csv(file_path, header=None, encoding='ISO-8859-1').values.flatten()
    
    # 문자열을 숫자형으로 변환합니다. 변환할 수 없는 값은 NaN으로 처리합니다.
    measurement_values = pd.to_numeric(measurement_values, errors='coerce')
    
    # 기준값과 편차 계산
    deviations = measurement_values - standard_value
    
    # 결과 데이터프레임 생성
    df = pd.DataFrame({
        'Standard Value': np.full(measurement_values.shape, standard_value),
        'Measurement Value': measurement_values,
        'Deviation': deviations,
        'Upper Tolerance': np.full(measurement_values.shape, upper_tolerance),
        'Lower Tolerance': np.full(measurement_values.shape, lower_tolerance)
    })
    return df

def main():
    # 데이터셋 경로 설정
    dataset_path = os.getcwd() + "\\dataset_csv\\45926-4G100"
    datalist = os.listdir(dataset_path)
    
    # 최종 데이터프레임 초기화 (처리된 데이터 저장용)
    final_df = pd.DataFrame()
    
    # 기준값, 상한 허용오차, 하한 허용오차 설정
    standard_value = 1.0
    upper_tolerance = 0.3
    lower_tolerance = -0.3
    
    # 각 파일에 대해 데이터 처리
    for file in datalist:
        file_path = os.path.join(dataset_path, file)
        df = process_file(file_path, standard_value, upper_tolerance, lower_tolerance)
        final_df = pd.concat([final_df, df], ignore_index=True)
    
    # 처리된 데이터프레임 저장
    final_df.to_csv(dataset_path + '\\sampling8.csv', index=False)

if __name__ == "__main__":
    main()