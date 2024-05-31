import pandas as pd
import numpy as np

# 파일 경로 설정
ok_file_path = "C:\\git_folder\\CMM_DeepLearning_Module\\MLP\\datas\\OK.csv"
ng_file_path = "C:\\git_folder\\CMM_DeepLearning_Module\\MLP\\datas\\NG.csv"

# 데이터프레임 읽기
ok_df = pd.read_csv(ok_file_path, encoding='cp949')
ng_df = pd.read_csv(ng_file_path, encoding='cp949')

# 각 열의 최솟값과 최댓값 계산
ok_min = ok_df.iloc[:, 1:].min()
ok_max = ok_df.iloc[:, 1:].max()
ng_min = ng_df.iloc[:, 1:].min()
ng_max = ng_df.iloc[:, 1:].max()

# 최솟값 범위와 최댓값 범위 설정
min_range = np.minimum(ok_min, ng_min)
max_range = np.maximum(ok_max, ng_max)

# 부트스트랩 샘플링 함수 정의
def bootstrap_sample(min_val, max_val, size):
    return np.round(np.random.uniform(min_val, max_val, size), 3)

# 부트스트랩 샘플링 수행
bootstrapped_df = pd.DataFrame()

# 첫 번째 열을 제외한 나머지 열에 대해 부트스트랩 샘플링 수행
for column in ok_df.columns[1:]:  # 첫 열 제외
    min_val = min_range[column]
    max_val = max_range[column]

    bootstrapped_samples = bootstrap_sample(min_val, max_val, size=1000)
    bootstrapped_df[column] = bootstrapped_samples

# 첫 번째 열을 동일하게 추가 (여기서는 NG 데이터프레임의 첫 번째 열을 사용)
bootstrapped_df.insert(0, ng_df.columns[0], ng_df[ng_df.columns[0]].iloc[:1000])

# 결과를 CSV 파일로 저장
output_path = "C:\\git_folder\\CMM_DeepLearning_Module\\MLP\\test\\ONLY_NG.csv"
bootstrapped_df.to_csv(output_path, index=False, encoding='cp949')