import pandas as pd
import numpy as np

ok_file_path = "C:\\git_folder\\CMM_DeepLearning_Module\\MLP\\datas\\OK.csv"
ng_file_path = "C:\\git_folder\\CMM_DeepLearning_Module\\MLP\\datas\\NG.csv"

ok_df = pd.read_csv(ok_file_path, encoding='cp949')
ng_df = pd.read_csv(ng_file_path, encoding='cp949')

ok_min = ok_df.iloc[:, 1:].min()
ok_max = ok_df.iloc[:, 1:].max()
ng_min = ng_df.iloc[:, 1:].min()
ng_max = ng_df.iloc[:, 1:].max()

min_range = np.minimum(ok_min, ng_min)
max_range = np.maximum(ok_max, ng_max)

def bootstrap_sample(min_val, max_val, size):
    return np.round(np.random.uniform(min_val, max_val, size), 3)

bootstrapped_df = pd.DataFrame()

for column in ok_df.columns[1:]:  # 첫 열 제외
    min_val = min_range[column]
    max_val = max_range[column]

    bootstrapped_samples = bootstrap_sample(min_val, max_val, size=1000)
    bootstrapped_df[column] = bootstrapped_samples
    
bootstrapped_df.insert(0, ng_df.columns[0], ng_df[ng_df.columns[0]].iloc[:1000])

output_path = "C:\\git_folder\\CMM_DeepLearning_Module\\MLP\\test\\ONLY_NG.csv"
bootstrapped_df.to_csv(output_path, index=False, encoding='cp949')