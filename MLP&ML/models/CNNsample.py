import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 데이터 전처리 및 준비
def prepare_data(csv_file):
    # CSV 파일을 읽어들여 DataFrame으로 변환
    all_data = pd.read_csv(csv_file, encoding='cp949')

    # 특징 선택 (불필요한 열 제거 등)
    selected_features = all_data.drop(columns=['품질상태'])  # 품질상태를 제외한 특징 선택

    # 숫자 데이터만 선택
    numeric_features = selected_features.select_dtypes(include=[np.number])

    # 딥러닝의 입력 데이터와 정답 데이터 생성
    X = numeric_features.values  # 입력 데이터
    y = all_data['품질상태'].values  # 출력 데이터

    # 테스트 데이터와 트레이닝 데이터로 분할
    X_train, X_test_full, Y_train, Y_test_full = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test, X_val, Y_test, Y_val = train_test_split(X_test_full, Y_test_full, test_size=0.5, random_state=42)

    # 데이터 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled= scaler.transform(X_val)

    # CNN에 맞게 2D로 형식 변경 (예: 7x1x1, 하나의 채널을 가진 2D 형태로 변환)
    X_train_scaled = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1, 1)
    X_test_scaled = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1, 1)
    X_val_scaled = X_val_scaled.reshape(-1, X_val_scaled.shape[1], 1, 1)

    return X_train_scaled, X_test_scaled, X_val_scaled, Y_train, Y_test, Y_val, scaler, selected_features.columns

def main():
    # CSV 파일 경로
    csv_file = 'C:\\Users\\freeman\\Desktop\\빅브라더\\MLP&ML\\datas\\data_jd_hd_delete_material_no_NTC_pca_component_7.csv'

    # 데이터 전처리
    X_train, X_test, X_val , y_train, y_test, Y_val, scaler, feature_columns = prepare_data(csv_file)

    # 특징과 레이블을 TensorFlow Dataset으로 변환
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(32)

    # Early Stopping 설정
    early_stopping = EarlyStopping(
    min_delta=0.001,  # 최소한의 변화
    patience=50,      # 몇 번 연속으로 개선이 없는지
    restore_best_weights=True  # 최상의 가중치로 복원
    )

    # CNN 모델 구성
    model = models.Sequential([
    layers.Conv2D(16, (2, 1), activation='relu', input_shape=(X_train.shape[1], 1, 1)),  # 첫 번째 합성곱층
    layers.MaxPooling2D(pool_size=(2, 1)),  # 풀링층
    layers.Conv2D(32, (2, 1), activation='relu'),  # 두 번째 합성곱층
    layers.MaxPooling2D(pool_size=(2, 1)),  # 풀링층
    layers.Flatten(),  # 1D 벡터로 변환
    layers.Dense(16, activation='relu'),  # 완전 연결층
    layers.Dense(1, activation='sigmoid')  # 출력층 (이진 분류)
    ])

    # 모델 컴파일
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 모델 학습
    history = model.fit(train_dataset, epochs=100, callbacks=[early_stopping], validation_data=val_dataset)

    # 모델 평가
    loss, accuracy = model.evaluate(test_dataset)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

    # 예측 결과
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # 성능 지표
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f'Accuracy: {acc}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')


# main 함수 실행
if __name__ == "__main__":
    main()