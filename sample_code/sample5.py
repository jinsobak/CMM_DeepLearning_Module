import pandas as pd
import numpy as np

데이터_수 = 10000
컬럼_수 = 5

np.random.seed(42)

데이터 = np.random.rand(데이터_수, 컬럼_수)

df = pd.DataFrame(데이터, columns=[f'컬럼{i+1}' for i in range(컬럼_수)])

샘플_df = df.sample(n=1000, random_state=42)

샘플_df.to_csv('sample_data.csv', index=False)

print("'sample_data.csv' 파일에 샘플 데이터프레임이 저장되었습니다.")