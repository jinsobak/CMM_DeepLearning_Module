###  cmd에서 이렇게 넣어서 실행해야함  streamlit run c:/git_folder/CMM_DeepLearning_Module/streamlit/app2.py
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Streamlit 앱 설정
st.title('딥러닝 품질 상태 예측')
st.write('CSV 파일을 업로드하여 품질 상태를 예측.')

# 데이터 전처리 및 준비
def prepare_data(csv_file):
    # CSV 파일을 읽어들여 DataFrame으로 변환
    all_data = pd.read_csv(csv_file, encoding='cp949')

    # 숫자 데이터만 선택
    numeric_features = all_data.select_dtypes(include=[np.number])
    
    # NaN 값을 가진 행 제거
    numeric_features = numeric_features.dropna()

    # 딥러닝의 입력 데이터와 정답 데이터 생성
    X = numeric_features.drop(columns=['품질상태']).values  # 입력 데이터
    y = numeric_features['품질상태'].values  # 출력 데이터

    # 데이터 스케일링
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, numeric_features.columns.drop('품질상태'), all_data

# CSV 파일 업로드 위젯
uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"], key="first_file")

if uploaded_file is not None:
    # 데이터 전처리
    X_train, X_test, y_train, y_test, scaler, numeric_features_columns, original_data = prepare_data(uploaded_file)

    # TensorFlow Dataset 생성
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

    # Early Stopping 설정
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=10,
        restore_best_weights=True
    )

    # Learning Rate 조절
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.0001
    )

    # 모델 구성
    def build_model(input_shape):
        model = models.Sequential([
            layers.Dense(512, activation='relu', input_shape=(input_shape,)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),

            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),

            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),

            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    model = build_model(X_train.shape[1])

    # 모델 학습
    with st.spinner('모델을 학습 중입니다...'):
        history = model.fit(
            train_dataset,
            epochs=100,
            callbacks=[early_stopping, reduce_lr],
            validation_data=test_dataset,
            verbose=0
        )

    # 모델 평가
    loss, accuracy = model.evaluate(test_dataset)
    st.write(f'**Test Loss:** {loss:.4f}, **Test Accuracy:** {accuracy:.4f}')

    # 예측 결과
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    st.write("**Confusion Matrix:**")
    st.write(cm)

    # 성능 지표
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.write(f'**Accuracy:** {acc:.4f}')
    st.write(f'**Precision:** {precision:.4f}')
    st.write(f'**Recall:** {recall:.4f}')
    st.write(f'**F1 Score:** {f1:.4f}')

    # 학습 결과 시각화
    def plot_history(history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(len(acc))

        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        st.pyplot(plt)

    plot_history(history)

    # 새로운 CSV 파일 업로드 기능 추가
st.write("### 새로운 CSV 파일을 업로드하여 이상치를 탐지하세요.")
new_file = st.file_uploader("새로운 CSV 파일 업로드", type=["csv"], key="new_file")

if new_file is not None:
    # 새로운 데이터 전처리
    new_data = pd.read_csv(new_file, encoding='cp949')
    numeric_features_new = new_data.select_dtypes(include=[np.number])
    numeric_features_new = numeric_features_new.dropna()

    # 기존 학습된 데이터와 동일한 열만 선택
    numeric_features_new = numeric_features_new[numeric_features_columns]

    X_new = numeric_features_new.values
    X_new_scaled = scaler.transform(X_new)

    # 재구성 오류 계산 (새로운 데이터의 이상치 탐지)
    y_new_pred_prob = model.predict(X_new_scaled)
    
    # 각 데이터 포인트의 재구성 오류를 계산하고, 평균 재구성 오류를 기준으로 이상치 판단
    reconstruction_error_new = np.mean(np.abs(X_new_scaled - model.predict(X_new_scaled)), axis=1)
    
    # 이상치 인덱스 탐지 (중복 제거)
    threshold = np.mean(reconstruction_error_new) + 2 * np.std(reconstruction_error_new)
    new_anomalies = np.where(reconstruction_error_new > threshold)[0]

    # 중복 제거된 이상치 인덱스 출력 (첫 번째 열 값 출력)
    st.write("**새로운 데이터의 이상치 인덱스 (첫 번째 열 값):**")
    anomaly_indices = new_data.iloc[new_anomalies].iloc[:, 0].values  # 첫 번째 열 값
    st.write(np.unique(anomaly_indices))
else:
    st.write("CSV 파일 업로드")
