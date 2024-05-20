import numpy as np
import pandas as pd

# 데이터 샘플링 함수 정의
def sample_data(num_samples=10):
    측정값_평면1_평면도 = np.random.uniform(0, 0.004, num_samples)
    기준값_평면1_평면도 = np.full(num_samples, 0.1)
    상한공차_평면1_평면도 = np.full(num_samples, 0.1)
    하한공차_평면1_평면도 = np.full(num_samples, 0)
    편차_평면1_평면도 = 기준값_평면1_평면도 - 측정값_평면1_평면도

    측정값_원1(I) <상>_D = np.random.uniform(16.465 , 16.509, num_samples)
    기준값_원1(I) <상>_D = np.full(num_samples, 0.1)
    상한공차_원1(I) <상>_D = np.full(num_samples, 0.1)
    하한공차_원1(I) <상>_D = np.full(num_samples, 0)
    편차_원1(I) <상>_D = 기준값_평면1_평면도 - 측정값_평면1_평면도
    
    return 측정값_평면1_평면도, 기준값_평면1_평면도, 상한공차_평면1_평면도, 하한공차_평면1_평면도, 편차_평면1_평면도

# 데이터 샘플링
num_samples = 10
측정값_평면1_평면도1, 기준값_평면1_평면도, 상한공차_평면1_평면도, 하한공차_평면1_평면도, 편차_평면1_평면도 = sample_data(num_samples)

# 데이터 프레임 생성
df = pd.DataFrame({
    '측정값_평면1_평면도': 측정값_평면1_평면도1,
    '기준값_평면1_평면도': 기준값_평면1_평면도,
    '상한공차_평면1_평면도': 상한공차_평면1_평면도,
    '하한공차_평면1_평면도': 하한공차_평면1_평면도,
    '편차_평면1_평면도': 편차_평면1_평면도
})

# 데이터 CSV 파일로 저장
file_path = "C:\\git_folder\\CMM_DeepLearning_Module\\MLP\\test\\sample_data3.csv"
df.to_csv(file_path, index=False, encoding='utf-8-sig')

print("데이터가 성공적으로 저장되었습니다.")