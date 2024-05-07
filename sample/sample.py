import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 데이터 전처리 및 준비
def prepare_data(csv_file):
    # CSV 파일을 읽어들여 DataFrame으로 변환
    all_data = pd.read_csv(csv_file, encoding='cp949')

    # 딥러닝의 입력 데이터와 정답 데이터 생성
    X = all_data.drop(columns=['품질상태', '품명']).values  # 입력 데이터
    y = all_data['품질상태'].values  # 출력 데이터

    # 테스트 데이터와 트레이닝 데이터로 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 데이터 스케일링
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)

    return X_train, X_test, y_train, y_test


# CSV 파일 경로
# or C:\\Users\\freeman\\Desktop\\빅브라더\\sample\\test_ld.csv
# C:\\Users\\ddc4k\\OneDrive\\Desktop\\빅브라더\\sample\\test_sd.csv
csv_file = "C:\\Users\\ddc4k\\OneDrive\\Desktop\\빅브라더\\sample\\data_mv,sv,dv_hd.csv"

# 데이터 전처리
X_train, X_test, y_train, y_test = prepare_data(csv_file)

# 특징과 레이블을 TensorFlow Dataset으로 변환합니다.
train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)).shuffle(len(X_train)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

# 딥러닝 모델 구성
model = models.Sequential([
    layers.Dense(64, activation='relu',
                 input_shape=(X_train.shape[1],)),  # 입력층
    layers.Dense(64, activation='relu'),  # 은닉층
    layers.Dense(1, activation='sigmoid')  # 출력층 (이진 분류)
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 학습
model.fit(train_dataset, epochs=100)

# 모델 평가
loss, accuracy = model.evaluate(test_dataset)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# 불량 판별
# 새로운 데이터를 모델에 입력하여 불량 여부를 판별합니다.
new_data = [
    # 새로운 데이터셋을 리스트로 저장
]
new_data = scalar.transform(new_data)  # 새로운 데이터 전처리
predictions = model.predict(new_data)
for i, pred in enumerate(predictions):
    if pred >= 0.5:
        print(f'데이터 {i + 1}은(는) 불량입니다.')
    else:
        print(f'데이터 {i + 1}은(는) 정상입니다.')
