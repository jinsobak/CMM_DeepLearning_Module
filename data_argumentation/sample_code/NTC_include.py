import pandas as pd
import numpy as np

file_path = "C:\\git_folder\\CMM_DeepLearning_Module\\MLP\\datas\\data_mv_sv_dv_ut_lt_hd.csv"

try:
    df = pd.read_csv(file_path, encoding='cp949')
except UnicodeDecodeError:
    print("error")

# 마지막 제외
df = df.iloc[:, :-1]

bootstrapped_df = pd.DataFrame()

for column in df.columns[1:]:  # 첫 열 제외
    min_val = df[column].min()
    max_val = df[column].max()

    bootstrapped_samples = np.random.uniform(low=min_val, high=max_val, size=1000)
    bootstrapped_samples = np.round(bootstrapped_samples, 3)
    bootstrapped_df[column] = bootstrapped_samples

bootstrapped_df.insert(0, df.columns[0], df[df.columns[0]].iloc[:1000])

output_path = "C:\\git_folder\\CMM_DeepLearning_Module\\MLP\\test\\bootstrapped_data.csv"
bootstrapped_df.to_csv(output_path, index=False, encoding='cp949')
