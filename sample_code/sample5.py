# 필요한 라이브러리를 불러옵니다.
import pandas as pd
import numpy as np

# 데이터와 컬럼의 수를 설정합니다.
데이터_수 = 10000
컬럼_수 = 5

# 재현 가능한 결과를 위해 시드를 설정합니다.
np.random.seed(42)

# 무작위 데이터를 생성합니다.
데이터 = np.random.rand(데이터_수, 컬럼_수)

# 판다스 데이터프레임을 생성합니다.
df = pd.DataFrame(데이터, columns=[f'컬럼{i+1}' for i in range(컬럼_수)])

# 데이터프레임에서 무작위로 1000개의 샘플을 추출합니다.
샘플_df = df.sample(n=1000, random_state=42)

# 샘플 데이터프레임을 'sample_data.csv' 파일로 저장합니다.
샘플_df.to_csv('sample_data.csv', index=False)

# 저장 완료 메시지를 출력합니다.
print("'sample_data.csv' 파일에 샘플 데이터프레임이 저장되었습니다.")