import pandas as pd
import os
import numpy as np

def augment_or_reduce_data(df, target_size=1000):
    current_size = len(df)
    if current_size == target_size:
        return df
    elif current_size < target_size:
        additional_rows_needed = target_size - current_size
        additional_rows = df.sample(n=additional_rows_needed, replace=True)  # replace=True allows for sampling the same row more than once.
        augmented_df = pd.concat([df, additional_rows], ignore_index=True)
        return augmented_df
    else:  # current_size > target_size
        reduced_df = df.sample(n=target_size)
        return reduced_df

def process_and_save_files(datasetPath, outputPath):
    datalist = os.listdir(datasetPath)
    combined_df = pd.DataFrame()

    for file in datalist:
        data = pd.read_csv(os.path.join(datasetPath, file), encoding='cp949')
        data = augment_or_reduce_data(data)
        combined_df = pd.concat([combined_df, data], ignore_index=True)

    os.makedirs(outputPath, exist_ok=True)
    combined_df.to_csv(path_or_buf=os.path.join(outputPath, "sampling4.csv"), encoding='cp949', index=False)

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    datasetPath = os.getcwd() + "\\output_test_ld\\45926-4G100"
    output_path = os.getcwd() + "\\MLP\\test"
    process_and_save_files(datasetPath, output_path)