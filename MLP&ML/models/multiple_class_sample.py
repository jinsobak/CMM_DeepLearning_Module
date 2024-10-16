import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 데이터 스케일링
    scalar = StandardScaler()
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.transform(X_test)

    # 레이블 원-핫 인코딩
    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)

    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, scalar

# 데이터 전처리
X_train, X_test, y_train, y_test, scalar = prepare_data('C:\\Users\\freeman\\Desktop\\빅브라더\\sample\\data_jd_hd.csv')

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
    layers.Dense(64, activation='relu',
                 input_shape=(X_train.shape[1],)),  # 입력층
    layers.Dense(32, activation='relu'),  # 은닉층
    layers.Dense(16, activation='relu'),  # 은닉층 추가
    layers.Dense(3, activation='softmax')  # 출력층 (다중 클래스 분류)
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

