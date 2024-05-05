import pandas as pd
import os
import numpy as np
 ##
def fill_or_drop_deviation(datas):
    datas.drop(datas[datas['항목'] == 'SMmf'].index, inplace=True)
    datas['편차'] = datas.apply(lambda x: float(x['기준값']) - float(x['측정값']) if x['편차'] == '-' and x['판정'] != '-' else x['편차'], axis=1)
    datas.drop(datas[datas['판정']=='-'].index, inplace=True)
    datas = datas[['편차', '기준값', '판정']]
    return datas

def augment_data(df, target_size=1000):
    current_size = len(df)
    if current_size >= target_size:
        return df.sample(n=target_size)  # 데이터가 충분히 많으면 샘플링하여 반환
    else:
        augmented_df = df
        while len(augmented_df) < target_size:
            diff = min(target_size - len(augmented_df), len(df))
            samples = df.sample(n=diff)
            augmented_df = pd.concat([augmented_df, samples], ignore_index=True)
        return augmented_df

if __name__=="__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('mode.chained_assignment', None)

    datasetPath = os.getcwd() + "\\output_test_ld\\45926-4G100"
    datalist = os.listdir(datasetPath)

    dataFrame = pd.DataFrame()
    for file in datalist:
        data = pd.read_csv(os.path.join(datasetPath, file), encoding='cp949')
        datas = pd.DataFrame(data)
        datas = fill_or_drop_deviation(datas)
        dataFrame = pd.concat([dataFrame, datas], ignore_index=True)

    # 데이터 증강
    dataFrame = augment_data(dataFrame, 1000)

    output_path = os.getcwd() + "\\MLP\\test"
    os.makedirs(output_path, exist_ok=True)  # 이미 존재하면 넘어감
    
    dataFrame.to_csv(path_or_buf=os.path.join(output_path, "sampleing.csv"), encoding='cp949')