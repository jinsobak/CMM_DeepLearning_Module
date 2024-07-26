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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 데이터 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, selected_features.columns

# CSV 파일 경로
csv_file = "C:\\Users\\freeman\\Desktop\\빅브라더\\sample\\data_mv_sv_dv_ut_lt_pca_no_NTC.csv"

# 데이터 전처리
X_train, X_test, y_train, y_test, scaler, feature_columns = prepare_data(csv_file)

# 특징과 레이블을 TensorFlow Dataset으로 변환합니다.
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(32)  # 배치 크기 조정
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)  # 배치 크기 조정

# Early Stopping 설정
early_stopping = EarlyStopping(
    min_delta=0.001,  # 최소한의 변화
    patience=50,      # 몇 번 연속으로 개선이 없는지
    restore_best_weights=True  # 최상의 가중치로 복원
)

# 딥러닝 모델 구성
model = models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),  # 첫 번째 은닉층
    layers.Dropout(0.3),  # Dropout 레이어 추가
    layers.Dense(256, activation='relu'),  # 두 번째 은닉층
    layers.Dropout(0.3),  # Dropout 레이어 추가
    layers.Dense(128, activation='relu'),  # 세 번째 은닉층
    layers.Dropout(0.3),  # Dropout 레이어 추가
    layers.Dense(64, activation='relu'),   # 네 번째 은닉층
    layers.Dropout(0.3),  # Dropout 레이어 추가
    layers.Dense(32, activation='relu'),   # 다섯 번째 은닉층
    layers.Dense(1, activation='sigmoid')  # 출력층 (이진 분류)
])

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model.fit(train_dataset, epochs=100, callbacks=[early_stopping], validation_data=test_dataset)

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

# 예측한 값과 실제 값이 일치하는 데이터와 일치하지 않는 데이터를 출력합니다.
correct_indices = np.where(y_test == y_pred.ravel())[0]
incorrect_indices = np.where(y_test != y_pred.ravel())[0]

print("\n맞춘 데이터 (정상/불량):")
for i in correct_indices:
    print(f"특징: {feature_columns}")
    print(f"데이터: {X_test[i]}")
    print(f"실제 값: {y_test[i]}, 예측 값: {y_pred[i]}")

print("\n틀린 데이터 (정상/불량):")
for i in incorrect_indices:
    print(f"특징: {feature_columns}")
    print(f"데이터: {X_test[i]}")
    print(f"실제 값: {y_test[i]}, 예측 값: {y_pred[i]}")
