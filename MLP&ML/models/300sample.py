import pandas as pd

# CSV 파일 읽기
input_file = "C:\\Users\\freeman\\Desktop\\빅브라더\\sample\\data_mv_sv_dv_ut_lt_hd_only_OK.csv"  # 원본 CSV 파일 경로
output_file = 'data_mv_sv_dv_ut_lt_hd_only_OK_sampleing.csv'  # 300줄만 남긴 후 저장할 CSV 파일 경로

df = pd.read_csv(input_file, encoding='cp949')  # 또는 encoding='cp949'를 시도해보세요

# 첫 300줄만 남기기
df_300 = df.head(180)

# 새로운 CSV 파일로 저장
df_300.to_csv(output_file, index=False, encoding='cp949')

print(f"First 180 rows have been saved to {output_file}")
