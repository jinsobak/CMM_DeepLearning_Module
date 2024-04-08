import pandas as pd
import os

def extract_dataframe_from_file(file_path):
    # Read the file with 'cp949' encoding
    with open(file_path, "r", encoding='cp949') as file:
        lines = file.readlines()

    # Initialize empty list to store the rows
    data = []

    # Extract Header info
    header_info = {
        "품명": lines[1].split("품    명:")[1].split("품    번:")[0].strip(),
        "품번": lines[1].split("품    번:")[1].strip(),
        "측정시간": lines[2].split("측정시간:")[1].split("측 정 자:")[0].strip(),
        "측정자": lines[2].split("측 정 자:")[1].strip(),
        "특기사항": lines[3].split(":")[1].strip(),
        "검사형태": lines[3].split("_")[1].strip(),
        "검사시간대": lines[3].split("_")[2].strip()+"간",
        "종믈검사": lines[3].split("_")[3].strip()+"물",
        "품질상태": lines[3].split("_")[-1].strip()  
    }
    if header_info['품질상태'] == '':
        header_info['품질상태'] = "NTC" # Need to Check

    # Iterate through the lines to extract the data
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Check if the line contains 번호 and 도형
        if line and line[0].isdigit():
            number, shape = line.split(maxsplit=1)

            # Read the next lines until an empty line or another header is found
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].strip()[0].isdigit():
                parts = lines[i].split()

                # Extract the data values with generalized handling for missing data
                if len(parts) >= 3:
                    item = parts[0]
                    if item == '평면도':
                        measured_value = parts[1]
                        standard_value = parts[2]
                        upper_tolerance = '-'
                        lower_tolerance = '-'
                        deviation = '-'
                        judgement = parts[-1]
                    elif item == 'SMmf':
                        measured_value = parts[1]
                        standard_value = parts[2]
                        upper_tolerance = parts[3]
                        lower_tolerance = parts[4]
                        deviation = parts[-1]
                        judgement = '-'
                    elif item == '원통도' :
                        measured_value = parts[1]
                        standard_value = parts[2]
                        upper_tolerance = '-'
                        lower_tolerance = '-'
                        deviation = '-'
                        judgement = '-'
                    elif item == '직각도':
                        measured_value = parts[1]
                        standard_value = parts[2]
                        upper_tolerance = '-'
                        lower_tolerance = '-'
                        deviation = '-'
                        judgement = parts[-1]
                    elif item == '동심도':
                        measured_value = parts[1]
                        standard_value = parts[2]
                        upper_tolerance = '-'
                        lower_tolerance = '-'
                        deviation = '-'
                        judgement = '-'
                    elif item == '평행도':
                        measured_value = parts[1]
                        standard_value = parts[2]
                        upper_tolerance = '-'
                        lower_tolerance = '-'
                        deviation = '-'
                        judgement = parts[-1]
                    else:
                        measured_value = parts[1]
                        standard_value = parts[2]
                        upper_tolerance = parts[3]
                        lower_tolerance = parts[4]
                        deviation = parts[5]
                        judgement = parts[-1]

                    row = [header_info['품명'],header_info['품번'],header_info['측정시간'],
                           header_info['측정자'],header_info['검사형태'],header_info['검사시간대'],
                           header_info['종믈검사'], number, shape,
                           item, measured_value, standard_value,
                           upper_tolerance, lower_tolerance, deviation,
                           judgement,header_info['품질상태']]
                    data.append(row)

                i += 1
        else:
            i += 1

    # Convert the list of rows to a DataFrame
    df = pd.DataFrame(data, columns=[
        "품명", "품번", "측정시간",
        "측정자", "검사형태", "검사시간대",
        "종믈검사", "번호", "도형",
        "항목", "측정값", "기준값",
        "상한공차", "하한공차", "편차",
        "판정","품질상태"
        ])

    return df

if __name__ == "__main__":
    dataset_path = os.getcwd() + "\\datasets"
    data_list = os.listdir(dataset_path)

    df_test = extract_dataframe_from_file(dataset_path+"\\"+data_list[2])

    df_test.to_csv(data_list[2]+".csv", index=False, encoding='cp949')

#%%
