import pandas as pd
import numpy as np
import os

def create_sample_data(num_samples=1000):
    # 임의의 데이터 생성을 위한 설정
    np.random.seed(0)
    product_names = ['Product_' + str(i) for i in range(1, 21)]  # 예시 품명
    shapes = ['Circle', 'Square', 'Triangle']  # 예시 도형
    items = ['ItemA', 'ItemB', 'ItemC']  # 예시 항목
    
    # 빈 데이터프레임 생성
    df_samples = pd.DataFrame()
    
    # 샘플 데이터 생성
    df_samples['품명'] = np.random.choice(product_names, num_samples)
    df_samples['도형'] = np.random.choice(shapes, num_samples)
    df_samples['항목'] = np.random.choice(items, num_samples)
    df_samples['측정값'] = np.random.uniform(0.5, 1.5, num_samples).round(2)
    df_samples['기준값'] = 1.0  # 기준값은 고정된 가정
    df_samples['편차'] = df_samples['측정값'] - df_samples['기준값']
    df_samples['판정'] = np.where(df_samples['편차'].abs() <= 0.2, 'OK', 'NG')
    df_samples['품질상태'] = np.random.choice([0, 1], num_samples)  # 0: NG, 1: OK
    
    return df_samples

if __name__ == "__main__":
    # 1000개의 샘플 데이터 생성
    df = create_sample_data(1000)
    
    # 데이터를 저장할 폴더 경로 설정 (현재 작업 디렉토리 내에 'MLP/datas' 폴더)
    output_path = os.path.join(os.getcwd(), "MLP", "test")
    
    # 폴더가 없다면 생성
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 데이터 샘플을 CSV 파일로 저장
    df.to_csv(os.path.join(output_path, "sample_data_1000.csv"), index=False, encoding='cp949')