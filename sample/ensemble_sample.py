import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 데이터 전처리 및 준비


def prepare_data(csv_file):
    all_data = pd.read_csv(csv_file, encoding='cp949')
    numeric_features = all_data.select_dtypes(include=[np.number])
    selected_features = numeric_features.drop(
        columns=['품질상태'])  # 품질상태를 제외한 특징 선택
    X = selected_features.values  # 입력 데이터
    y = all_data['품질상태'].values  # 출력 데이터
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoder

# 각 데이터셋에 대해 모델 학습


def train_model(csv_file, loss, activation):
    X_train, X_test, y_train, y_test, scaler, encoder = prepare_data(csv_file)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)).shuffle(len(X_train)).batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (X_test, y_test)).batch(32)
    early_stopping = EarlyStopping(
        min_delta=0.001, patience=20, restore_best_weights=True)
    model = models.Sequential([
        Input(shape=(X_train.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation=activation)  # 이진 분류를 위한 출력 레이어
    ])
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    model.fit(train_dataset, epochs=100, callbacks=[
              early_stopping], validation_data=test_dataset, verbose=1)
    loss, accuracy = model.evaluate(test_dataset)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
    return model, scaler, encoder


# CSV 파일 경로
# or C:\\Users\\ddc4k\\OneDrive\\Desktop\\빅브라더\\sample
ng_csv_file = "C:\\Users\\freeman\\Desktop\\빅브라더\\sample\\data_mv_sv_dv_ut_lt_hd_only_NG.csv"
ok_csv_file = "C:\\Users\\freeman\\Desktop\\빅브라더\\sample\\data_mv_sv_dv_ut_lt_hd_only_OK.csv"
with_ntc_csv_file = "C:\\Users\\freeman\\Desktop\\빅브라더\\sample\\data_mv_sv_dv_ut_lt_hd_no_NTC.csv"

# NG, OK 모델 학습
ng_model, ng_scaler, ng_encoder = train_model(
    ng_csv_file, 'binary_crossentropy', 'sigmoid')
ok_model, ok_scaler, ok_encoder = train_model(
    ok_csv_file, 'binary_crossentropy', 'sigmoid')

# 기본 모델의 예측 결합


def get_predictions(models, scalers, X):
    predictions = []
    for model, scaler in zip(models, scalers):
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)
        predictions.append(pred)
    return np.concatenate(predictions, axis=1)


# 모든 클래스가 포함된 데이터셋 준비
X_train_with_ntc, X_test_with_ntc, y_train_with_ntc, y_test_with_ntc, scaler_with_ntc, encoder_with_ntc = prepare_data(
    with_ntc_csv_file)

# 메타 모델 학습 데이터 생성
train_meta = get_predictions([ng_model, ok_model], [
                             ng_scaler, ok_scaler], X_train_with_ntc)
test_meta = get_predictions([ng_model, ok_model], [
                            ng_scaler, ok_scaler], X_test_with_ntc)

# 메타 모델 학습
meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(train_meta, np.argmax(y_train_with_ntc, axis=1))

# 메타 모델 예측 및 평가
meta_train_predictions = meta_model.predict(train_meta)
meta_test_predictions = meta_model.predict(test_meta)

train_accuracy = accuracy_score(
    np.argmax(y_train_with_ntc, axis=1), meta_train_predictions)
test_accuracy = accuracy_score(
    np.argmax(y_test_with_ntc, axis=1), meta_test_predictions)

print(f"Meta Train Accuracy: {train_accuracy}")
print(f"Meta Test Accuracy: {test_accuracy}")

# 새로운 데이터 예측 함수


def predict_new_data(models, scalers, meta_model, new_data):
    base_predictions = get_predictions(models, scalers, new_data)
    final_predictions = meta_model.predict(base_predictions)
    return final_predictions


# 불량 판별
new_data = [
    # 새로운 데이터셋을 리스트로 저장
]

if len(new_data) == 0:
    print("새로운 데이터가 없습니다.")
else:
    new_data_scaled = scaler_with_ntc.transform(new_data)  # 새로운 데이터 전처리
    predictions = predict_new_data(
        [ng_model, ok_model], [ng_scaler, ok_scaler], meta_model, new_data_scaled)
    for i, pred in enumerate(predictions):
        if pred == 0:
            print(f'데이터 {i + 1}은(는) 불량입니다.')
        elif pred == 1:
            print(f'데이터 {i + 1}은(는) 정상입니다.')
        elif pred == 2:
            print(f'데이터 {i + 1}은(는) 모름입니다.')
