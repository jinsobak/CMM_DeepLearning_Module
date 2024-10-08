import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping

# 데이터 전처리 및 준비
def prepare_data(csv_file):
    # CSV 파일을 읽어들여 DataFrame으로 변환
    all_data = pd.read_csv(csv_file, encoding='cp949')

    # 특징 선택 (불필요한 열 제거 등)
    # 숫자형 데이터만 선택
    numeric_features = all_data.select_dtypes(include=[np.number])
    selected_features = numeric_features.drop(columns=['품질상태'])  # 품질상태를 제외한 특징 선택

    # 딥러닝의 입력 데이터와 정답 데이터 생성
    X = selected_features.values  # 입력 데이터
    y = all_data['품질상태'].values  # 출력 데이터

    # 원-핫 인코딩
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y.reshape(-1, 1))

    # 테스트 데이터와 트레이닝 데이터로 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 데이터 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoder


# CSV 파일 경로
csv_file = "C:\\Users\\freeman\\Desktop\\빅브라더\\sample\\\data_mv,sv,dv_hd_with_NTC.csv"

# 데이터 전처리
X_train, X_test, y_train, y_test, scaler, encoder = prepare_data(csv_file)

# 클래스 수 확인
num_classes = y_train.shape[1]
print(f"Number of classes: {num_classes}")

# 특징과 레이블을 TensorFlow Dataset으로 변환합니다.
train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)).shuffle(len(X_train)).batch(32)  # 배치 크기 조정
test_dataset = tf.data.Dataset.from_tensor_slices(
    (X_test, y_test)).batch(32)  # 배치 크기 조정

# Early Stopping 설정
early_stopping = EarlyStopping(
    min_delta=0.001,  # 최소한의 변화
    patience=20,      # 몇 번 연속으로 개선이 없는지
    restore_best_weights=True  # 최상의 가중치로 복원
)

# 딥러닝 모델 구성
model = models.Sequential([
    # input / encoder
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),

    # Decoder
    layers.Dense(16, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(64, activation='relu'),
    # 활성화 함수를 softmax로 변경하고 클래스 수에 맞춰 출력 레이어 수정
    layers.Dense(num_classes, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
model.fit(train_dataset, epochs=100, callbacks=[
          early_stopping], validation_data=test_dataset)

# 모델 평가
loss, accuracy = model.evaluate(test_dataset)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# 불량 판별
# 새로운 데이터를 모델에 입력하여 불량 여부를 판별합니다.
new_data = [
    # 새로운 데이터셋을 리스트로 저장
]

if len(new_data) == 0:
    print("새로운 데이터가 없습니다.")
else:
    new_data_scaled = scaler.transform(new_data)  # 새로운 데이터 전처리
    predictions = model.predict(new_data_scaled)
    for i, pred in enumerate(predictions):
        predicted_class = np.argmax(pred)
        if predicted_class == 1:  # 예를 들어, 클래스 1이 불량이라고 가정
            print(f'데이터 {i + 1}은(는) 불량입니다.')
        else:
            print(f'데이터 {i + 1}은(는) 정상입니다.')
