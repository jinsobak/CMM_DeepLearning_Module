import numpy as np
import pandas as pd

# 데이터 샘플링 함수 정의
def sample_data(num_samples=10):
    측정값_평면1 = np.random.uniform(0, 0.004, num_samples)
    기준값_평면1 = np.full(num_samples, 0.1)
    상한공차_평면1 = np.full(num_samples, 0.1)
    하한공차_평면1 = np.full(num_samples, 0)
    편차_평면1 = 기준값_평면1 - 측정값_평면1  # 편차를 기준값 - 측정값으로 계산
    return 측정값_평면1, 기준값_평면1, 상한공차_평면1, 하한공차_평면1, 편차_평면1

# 데이터 샘플링
num_samples = 10
측정값_평면1, 기준값_평면1, 상한공차_평면1, 하한공차_평면1, 편차_평면1 = sample_data(num_samples)

# 데이터 프레임 생성
df = pd.DataFrame({
    '측정값_평면1': 측정값_평면1,
    '기준값_평면1': 기준값_평면1,
    '상한공차_평면1': 상한공차_평면1,
    '하한공차_평면1': 하한공차_평면1,
    '편차_평면1': 편차_평면1
})

# 데이터 CSV 파일로 저장
file_path = "C:\\git_folder\\CMM_DeepLearning_Module\\MLP\\test\\sample_data2.csv"
df.to_csv(file_path, index=False, encoding='utf-8-sig')

print("데이터가 성공적으로 저장되었습니다.")