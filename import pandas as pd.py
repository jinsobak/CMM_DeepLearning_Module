import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 데이터 전처리 및 준비
def prepare_data(csv_files):
    # CSV 파일을 읽어들여 DataFrame으로 변환
    all_data = pd.read_csv(csv_files, encoding='cp949')
    print(all_data.columns)

    # 필요한 특징 추출 및 변환
    X = all_data.drop(columns=['품질상태'])  # 입력 데이터
    y = all_data['품질상태']  # 출력 데이터

    # 라벨 인코딩
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# 모델 구축
def build_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # 시그모이드 함수를 사용하여 확률값을 출력
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 데이터 파일 경로
csv_files = 'C:\\Users\\pc\\Desktop\\4학년1학기\\SW_project\\CMM_analysis\\test_ld.csv'

# 데이터 전처리 및 준비
X_train, X_test, y_train, y_test = prepare_data(csv_files)

# 모델 구축
input_dim = X_train.shape[1]
model = build_model(input_dim)

# 모델 학습
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_train, y_train))

# 모델 평가
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

prd = model.predict(X_test)
print(prd)

# 학습된 모델을 .h5 파일로 저장하는 함수
def save_model(model, file_path):
    model.save(file_path)
    print(f"모델이 {file_path}에 저장되었습니다.")

# 모델 저장
save_model(model, 'trained_model.h5')