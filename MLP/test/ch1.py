import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

def prepare_data(ng_file, ok_file, ntc_file):
    # CSV 파일을 읽어들여 DataFrame으로 변환
    ng_data = pd.read_csv(ng_file, encoding='cp949')
    ok_data = pd.read_csv(ok_file, encoding='cp949')
    ntc_data = pd.read_csv(ntc_file, encoding='cp949')

    # 라벨 추가
    ng_data['label'] = 0
    ok_data['label'] = 1
    ntc_data['label'] = 2

    return ng_data, ok_data, ntc_data

def preprocess_data(ng_data, ok_data, ntc_data, labels):
    # 선택된 데이터만 포함
    data = pd.concat([ng_data[ng_data['label'].isin(labels)],
                      ok_data[ok_data['label'].isin(labels)],
                      ntc_data[ntc_data['label'].isin(labels)]])
    
    # 특징 선택 (불필요한 열 제거 등)
    selected_features = data.drop(columns=['label'])  # label을 제외한 특징 선택

    # 숫자 데이터만 선택
    numeric_features = selected_features.select_dtypes(include=[np.number])

    # 딥러닝의 입력 데이터와 정답 데이터 생성
    X = numeric_features.values  # 입력 데이터
    y = data['label'].values  # 출력 데이터

    # 테스트 데이터와 트레이닝 데이터로 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 데이터 스케일링
    scalar = StandardScaler()
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def build_and_train_model(X_train, y_train, X_test, y_test):
    # Early Stopping 설정
    early_stopping = EarlyStopping(
        min_delta=0.001,  # 최소한의 변화
        patience=20,      # 몇 번 연속으로 개선이 없는지
        restore_best_weights=True  # 최상의 가중치로 복원
    )

    # 특징과 레이블을 TensorFlow Dataset으로 변환합니다.
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)).shuffle(len(X_train)).batch(32)  # 배치 크기 조정
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (X_test, y_test)).batch(32)  # 배치 크기 조정

    # 딥러닝 모델 구성
    model = models.Sequential([
        layers.Dense(64, activation='relu',
                     input_shape=(X_train.shape[1],)),  # 입력층
        layers.Dense(32, activation='relu'),  # 은닉층
        layers.Dense(16, activation='relu'),  # 은닉층 추가
        layers.Dense(1, activation='sigmoid')  # 출력층 (이진 분류)
    ])

    # 모델 컴파일
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 모델 학습
    model.fit(train_dataset, epochs=100, callbacks=[
              early_stopping], validation_data=test_dataset)

    # 모델 평가
    loss, accuracy = model.evaluate(test_dataset)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

    return model

# 파일 경로
ng_file = r"C:\git_folder\CMM_DeepLearning_Module\MLP\datas\data_mv_sv_dv_ut_lt_hd_only_NG.csv"
ok_file = r"C:\git_folder\CMM_DeepLearning_Module\MLP\datas\data_mv_sv_dv_ut_lt_hd_only_OK.csv"
ntc_file = r"C:\git_folder\CMM_DeepLearning_Module\MLP\datas\data_mv_sv_dv_ut_lt_hd_only_NTC.csv"

# 데이터 전처리
ng_data, ok_data, ntc_data = prepare_data(ng_file, ok_file, ntc_file)

# NG vs OK
X_train_ng_ok, X_test_ng_ok, y_train_ng_ok, y_test_ng_ok = preprocess_data(ng_data, ok_data, ntc_data, [0, 1])
model_ng_ok = build_and_train_model(X_train_ng_ok, y_train_ng_ok, X_test_ng_ok, y_test_ng_ok)

# NG vs NTC
#X_train_ng_ntc, X_test_ng_ntc, y_train_ng_ntc, y_test_ng_ntc = preprocess_data(ng_data, ok_data, ntc_data, [0, 2])
#model_ng_ntc = build_and_train_model(X_train_ng_ntc, y_train_ng_ntc, X_test_ng_ntc, y_test_ng_ntc)

# OK vs NTC
X_train_ok_ntc, X_test_ok_ntc, y_train_ok_ntc, y_test_ok_ntc = preprocess_data(ng_data, ok_data, ntc_data, [1, 2])
model_ok_ntc = build_and_train_model(X_train_ok_ntc, y_train_ok_ntc, X_test_ok_ntc, y_test_ok_ntc)

from sklearn.metrics import accuracy_score

# 최종 데이터 (테스트 데이터)
X_test_combined = np.concatenate([X_test_ng_ok, X_test_ok_ntc])
y_test_combined = np.concatenate([y_test_ng_ok, y_test_ok_ntc])

# 각 모델의 예측 결과
y_pred_ng_ok = (model_ng_ok.predict(X_test_combined) > 0.5).astype(int)
#y_pred_ng_ntc = (model_ng_ntc.predict(X_test_combined) > 0.5).astype(int)
y_pred_ok_ntc = (model_ok_ntc.predict(X_test_combined) > 0.5).astype(int)

# 앙상블 예측 결과 (다수결)
y_pred_ensemble = np.round((y_pred_ng_ok + y_pred_ok_ntc) / 3)

# 최종 정확도 평가
final_accuracy = accuracy_score(y_test_combined, y_pred_ensemble)
print(f'Final Ensemble Accuracy: {final_accuracy}')
