import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from deep_learning import prepare_data

def load_saved_model(file_path):                                                                                       # 저장된 모델 불러오기
    model = load_model(file_path)
    return model

def preprocess_input_data(csv_file, label_encoder):                                                                    # 입력 데이터 전처리
    input_data = pd.read_csv(csv_file, encoding='cp949')                                                               # CSV 파일을 DataFrame으로 변환
    
    X = input_data.drop(columns=['품질상태'])                                                                           # 품질상태 열을 제외한 입력 데이터 추출
    
    X_encoded = label_encoder.transform(X.values.flatten())                                                             # 라벨 인코딩 적용하고 2차원 배열을 1차원으로 평탄화
    return X_encoded

def make_predictions(model, input_data):                                                                                # 모델을 사용하여 예측 수행
    predictions = model.predict(input_data)                                            # 1차원 배열로 변환 후 예측 수행
    return predictions

csv_file_path ='C:\\Users\\pc\\Desktop\\4학년1학기\\SW_project\\CMM_analysis\\test_sd.csv' # 데이터 파일 경로
model_file_path = 'C:\\Users\\pc\\Desktop\\keras_data\\trained_model.keras'                                             # 저장된 모델 파일 경로

X_train, X_test, y_train, y_test = prepare_data(csv_file_path)
print(X_test.shape[1])

loaded_model = load_saved_model(model_file_path)                                                                        # 저장된 모델 불러오기

label_encoder = LabelEncoder()                                                                                          # 라벨 인코더 생성 및 fit 메서드 호출
label_encoder.fit(['정상', '불량'])                                                                                      # 라벨 인코더에 라벨을 명시적으로 지정

#X_input = preprocess_input_data(csv_file_path, label_encoder)                                                           # 입력 데이터 전처리

predictions = make_predictions(loaded_model, X_test)                                                                   # 예측 수행

print(predictions)                                                                                                      # 결과 출력