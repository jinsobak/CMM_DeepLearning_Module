import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def prepare_data(csv_files):                                                                                    # 데이터 전처리 및 준비
    all_data = pd.read_csv(csv_files, encoding='cp949')                                                         # CSV 파일을 읽어들여 DataFrame으로 변환
    print(all_data.columns)

    X = all_data.drop(columns=['품명', '품질상태'])                                                                      # 필요한 특징 추출 및 변환하고 입력 데이터
    y = all_data['품질상태']                                                                                     # 출력 데이터

    label_encoder = LabelEncoder()                                                                              # 라벨 인코딩
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)                   # Train-test split
    return X_train, X_test, y_train, y_test

def build_model(input_dim):                                                                                     # 모델 구축
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')                                                                          # 시그모이드 함수를 사용하여 확률값을 출력
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def save_model_native_keras(model, file_path):                                                                  # 네이티브 케라스 형식으로 모델 저장
    model.save(file_path)

if __name__=="__main__":
    csv_files = 'C:\\Users\\pc\\Desktop\\4학년1학기\\SW_project\\CMM_analysis\\test_sd.csv'                          # 데이터 파일 경로

    X_train, X_test, y_train, y_test = prepare_data(csv_files)                                                      # 데이터 전처리 및 준비

    input_dim = X_train.shape[1]   
    print(input_dim)                                                                                 # 모델 구축
    model = build_model(input_dim)

    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_train, y_train))                      # 모델 학습

    save_model_native_keras(model, 'C:\\Users\\pc\\Desktop\\keras_data\\trained_model.keras')                       # 네이티브 케라스 형식으로 모델 저장

    test_loss, test_accuracy = model.evaluate(X_test, y_test)                                                       # 모델 평가
    print("Test Accuracy:", test_accuracy,"Test Loss:",test_loss)


