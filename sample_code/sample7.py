import pandas as pd
import numpy as np
import os

# 샘플 데이터를 생성하는 함수
def create_sample_data(num_samples=1000):
    np.random.seed(0)  # 결과의 일관성을 위해 시드 설정

    # 데이터 샘플링을 위한 랜덤 값 생성
    measurement_values = np.random.normal(loc=1.0, scale=0.05, size=num_samples)  # 평균 1.0, 표준편차 0.05의 정규분포에서 측정값 샘플링
    standard_values = np.ones(num_samples)  # 기준값은 모두 1로 설정
    deviations = measurement_values - standard_values  # 편차 계산
    upper_tolerance = np.ones(num_samples) * 0.3  # 상한공차는 모두 0.3으로 설정
    lower_tolerance = np.ones(num_samples) * -0.3  # 하한공차는 모두 -0.3으로 설정

    # 데이터 프레임 생성
    df = pd.DataFrame({
        '기준값': standard_values,
        '측정값': measurement_values,
        '편차': deviations,
        '상한공차': upper_tolerance,
        '하한공차': lower_tolerance
    })

    # 생성된 데이터프레임 반환
    return df

# 데이터 샘플 생성 및 저장
df_sample = create_sample_data()

# 데이터 저장할 경로 확인 및 생성
save_path = 'MLP/test'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# csv 파일로 저장
df_sample.to_csv(f'{save_path}/sampling7.csv', index=False)