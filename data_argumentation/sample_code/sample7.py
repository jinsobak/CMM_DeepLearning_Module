import pandas as pd
import numpy as np
import os

def create_sample_data(num_samples=1000):
    np.random.seed(0) 

    measurement_values = np.random.normal(loc=1.0, scale=0.05, size=num_samples) 
    standard_values = np.ones(num_samples)  
    deviations = measurement_values - standard_values  
    upper_tolerance = np.ones(num_samples) * 0.3 
    lower_tolerance = np.ones(num_samples) * -0.3 

    df = pd.DataFrame({
        '기준값': standard_values,
        '측정값': measurement_values,
        '편차': deviations,
        '상한공차': upper_tolerance,
        '하한공차': lower_tolerance
    })

    return df

df_sample = create_sample_data()

save_path = 'MLP/test'
if not os.path.exists(save_path):
    os.makedirs(save_path)

df_sample.to_csv(f'{save_path}/sampling7.csv', index=False)