import pandas as pd
import numpy as np
import os

def create_sample_data(num_samples=1000):
    np.random.seed(0)
    product_names = ['Product_' + str(i) for i in range(1, 21)]
    shapes = ['Circle', 'Square', 'Triangle'] 
    items = ['ItemA', 'ItemB', 'ItemC'] 
    
    df_samples = pd.DataFrame()
    
    df_samples['품명'] = np.random.choice(product_names, num_samples)
    df_samples['도형'] = np.random.choice(shapes, num_samples)
    df_samples['항목'] = np.random.choice(items, num_samples)
    df_samples['측정값'] = np.random.uniform(0.5, 1.5, num_samples).round(2)
    df_samples['기준값'] = 1.0  
    df_samples['편차'] = df_samples['측정값'] - df_samples['기준값']
    df_samples['판정'] = np.where(df_samples['편차'].abs() <= 0.2, 'OK', 'NG')
    df_samples['품질상태'] = np.random.choice([0, 1], num_samples)  
    
    return df_samples

if __name__ == "__main__":
    df = create_sample_data(1000)
    
    output_path = os.path.join(os.getcwd(), "MLP", "test")
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df.to_csv(os.path.join(output_path, "sample_data_1000.csv"), index=False, encoding='cp949')