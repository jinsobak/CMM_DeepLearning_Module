import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# 학습된 모델을 불러옵니다
model = tf.keras.models.load_model('trained_model.keras')
print("모델이 'trained_model.keras' 파일에서 로드되었습니다.")

# 데이터를 전처리하는 함수입니다
def prepare_new_data(csv_file, scaler):
    # CSV 파일을 읽어들여 DataFrame으로 변환
    all_data = pd.read_csv(csv_file, encoding='cp949')

    # 숫자 데이터만 선택
    numeric_features = all_data.select_dtypes(include=[np.number])

    # 피처 수 확인
    print(f"새로운 데이터의 피처 수: {numeric_features.shape[1]}")
    
    # 딥러닝의 입력 데이터 생성
    X = numeric_features.values  # 입력 데이터

    # 데이터 스케일링
    X_scaled = scaler.transform(X)

    # 만약 레이블이 포함되어 있다면 레이블도 분리
    if '품질상태' in all_data.columns:
        y = all_data['품질상태'].values
        return X_scaled, y
    else:
        return X_scaled

# 스케일러를 생성하기 위해 원래의 데이터셋을 사용합니다
def prepare_scaler(csv_file):
    all_data = pd.read_csv(csv_file, encoding='cp949')
    selected_features = all_data.drop(columns=['품질상태'])
    numeric_features = selected_features.select_dtypes(include=[np.number])
    X = numeric_features.values
    scalar = StandardScaler()
    scalar.fit(X)
    return scalar

# 원래 학습에 사용된 데이터셋 경로를 설정합니다
original_csv_file = "C:\\Users\\freeman\\Desktop\\빅브라더\\sample\\data_mv_sv_dv_ut_lt_hd_no_NTC.csv"
scaler = prepare_scaler(original_csv_file)

# 새로운 데이터셋 경로를 설정합니다
new_csv_file = "C:\\Users\\freeman\\Desktop\\빅브라더\\sample\\NTC_include.csv"
new_data_result = prepare_new_data(new_csv_file, scaler)

# NTC(모름) 데이터를 포함한 새로운 데이터 예측
if isinstance(new_data_result, tuple):
    new_data, new_labels = new_data_result
    # 평가 (loss와 accuracy 출력)
    loss, accuracy = model.evaluate(new_data, new_labels)
    print(f'New Data Loss: {loss}, New Data Accuracy: {accuracy}')

    # 예측 수행
    predictions = model.predict(new_data)
else:
    new_data = new_data_result
    # 예측 수행
    predictions = model.predict(new_data)

# 예측 결과 디버깅 메시지 추가
print(f"예측 결과 (첫 10개): {predictions[:10]}")

# 예측 결과 출력
for i, pred in enumerate(predictions):
    if pred >= 0.5:
        print(f'데이터 {i + 1}은(는) 불량입니다.')
    else:
        print(f'데이터 {i + 1}은(는) 정상입니다.')

# 전체 예측 결과 출력 (디버깅용)
print(f"전체 예측 결과: {predictions}")
