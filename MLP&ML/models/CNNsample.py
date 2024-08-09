import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

# 데이터 전처리 및 준비
def prepare_data(csv_file):
    all_data = pd.read_csv(csv_file, encoding='cp949')
    X = all_data.drop(columns=['품질상태', '품명']).values
    y = all_data['품질상태'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scalar = StandardScaler()
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scalar

csv_file = "C:\\Users\\freeman\\Desktop\\빅브라더\\sample\\data_dv_hd.csv"
X_train, X_test, y_train, y_test, scalar = prepare_data(csv_file)

# 모델 구성
model = models.Sequential([
    layers.Reshape((X_train.shape[1], 1), input_shape=(X_train.shape[1],)),
    layers.Conv1D(64, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(128, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2)

# 모델 평가
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)
f1 = f1_score(y_test, y_pred_binary)
print(f'F1 Score: {f1}')
