import pandas as pd
import os

def extract_data_from_file(file_path, fileName):
    with open(file_path, 'r', encoding="cp949") as f:
        lines = f.readlines()
        
    header_info = {
            "품명": lines[1].split("품    명:")[1].split("품    번:")[0].strip(),
            "품번": lines[1].split("품    번:")[1].strip(),
            "측정시간": lines[2].split("측정시간:")[1].split("측 정 자:")[0].strip(),
            "측정자": lines[2].split("측 정 자:")[1].strip(),
            "특기사항": lines[3].split(":")[1].strip(),
            "검사형태": lines[3].split("_")[1].strip(),
            "검사시간대": lines[3].split("_")[2].strip()+"간",
            "종믈검사": lines[3].split("_")[3].strip()+"물",
            "품질상태": ""
    }
    feature = lines[3].split(":")[1].strip()+".txt"

    if(len(feature.split('_')) < 6):
        header_info["품질상태"] = ''
    else:
        header_info["품질상태"] = lines[3].split("_")[-1].strip()
    
    if header_info["품질상태"] == '':
        header_info["품질상태"] = "NTC"

    data = []
    
    i = 0
    while(i < len(lines)):
        line = lines[i].strip()
        if line and line[0].isdigit():
            #번호, 도형
            number, shape = line.split(maxsplit=1)
            i += 1
            while i < len(lines) and lines[i].strip() and \
                not lines[i].strip()[0].isdigit():
                    
                parts = lines[i].split()
                
                if(len(parts) >= 3):
                    item = parts[0]
                    if item == "평면도":
                        #측정값
                        measured_value = parts[1]
                        #기준값
                        standard_value = parts[2]
                        #상한공차
                        upper_tolerance = '-'
                        #하한공차
                        lower_tolerance = '-'
                        #편차
                        deviation = '-'
                        #판정
                        judgement = parts[-1]
                    elif item == "SMmf": #나중에 아예 뺄 수도 있음
                        measured_value = parts[1]
                        standard_value = parts[2]
                        upper_tolerance = parts[3]
                        lower_tolerance = parts[4]
                        deviation = parts[-1]
                        judgement = '-'
                    elif item == "원통도" or item == "동심도" or item == "진원도":
                        measured_value = parts[1]
                        standard_value = parts[2]
                        upper_tolerance = '-'
                        lower_tolerance = '-'
                        deviation = '-'
                        judgement = '-'
                    # elif item == "동심도":
                    #     measured_value = parts[1]
                    #     standard_value = parts[2]
                    #     upper_tolerance = '-'
                    #     lower_tolerance = '-'
                    #     deviation = '-'
                    #     judgement = '-'
                    elif item == "직각도" or item == "평행도":
                        measured_value = parts[1]
                        standard_value = parts[2]
                        upper_tolerance = '-'
                        lower_tolerance = '-'
                        deviation = '-'
                        judgement = parts[-1]
                    # elif item == "평행도":
                    #     measured_value = parts[1]
                    #     standard_value = parts[2]
                    #     upper_tolerance = '-'
                    #     lower_tolerance = '-'
                    #     deviation = '-'
                    #     judgement = parts[-1]
                    elif item == "위치도":
                        measured_value = parts[1]
                        standard_value = parts[2]
                        upper_tolerance = '-'
                        lower_tolerance = parts[3]
                        deviation = '-'
                        judgement = '-'
                    else:
                        measured_value = parts[1]
                        standard_value = parts[2]
                        upper_tolerance = parts[3]
                        lower_tolerance = parts[4]
                        deviation = parts[5]
                        judgement = parts[-1]
                    
                    row = [header_info["품명"], header_info["품번"], header_info["측정시간"],
                           header_info['측정자'], header_info['검사형태'], header_info["검사시간대"],
                           header_info["종믈검사"], number, shape,
                           item, measured_value, standard_value, 
                           upper_tolerance, lower_tolerance, deviation,
                           judgement, header_info["품질상태"]]
                    data.append(row)
                i += 1
        else: #번호 도형으로 이루어진 행이 아니라면
            i += 1

    df = pd.DataFrame(data, columns=[
        "품명", "품번", "측정시간", "측정자",
        "검사형태", "검사시간대", "종물검사", 
        "번호", "도형", "항목", "측정값",
        "기준값", "상한공차", "하한공차", "편차",
        "판정", "품질상태"
    ])

    #print(df.head)
    
    return df

if __name__ == "__main__":
    dataset_path = os.getcwd() + "\\csv_datas"
    data_list = os.listdir(dataset_path)
    print(data_list)
    
    df_test = extract_data_from_file(dataset_path+"\\"+data_list[8])
    
    #print(df_test)