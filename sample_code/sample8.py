import pandas as pd
import numpy as np
import os

def process_file(file_path, standard_value, upper_tolerance, lower_tolerance):
    measurement_values = pd.read_csv(file_path, header=None, encoding='ISO-8859-1').values.flatten()
    
    measurement_values = pd.to_numeric(measurement_values, errors='coerce')
    
    deviations = measurement_values - standard_value
    
    df = pd.DataFrame({
        'Standard Value': np.full(measurement_values.shape, standard_value),
        'Measurement Value': measurement_values,
        'Deviation': deviations,
        'Upper Tolerance': np.full(measurement_values.shape, upper_tolerance),
        'Lower Tolerance': np.full(measurement_values.shape, lower_tolerance)
    })
    return df

def main():
    dataset_path = os.getcwd() + "\\dataset_csv\\45926-4G100"
    datalist = os.listdir(dataset_path)

    final_df = pd.DataFrame()

    standard_value = 1.0
    upper_tolerance = 0.3
    lower_tolerance = -0.3

    for file in datalist:
        file_path = os.path.join(dataset_path, file)
        df = process_file(file_path, standard_value, upper_tolerance, lower_tolerance)
        final_df = pd.concat([final_df, df], ignore_index=True)
    
    final_df.to_csv(dataset_path + '\\sampling8.csv', index=False)

if __name__ == "__main__":
    main()