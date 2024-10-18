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
st.set_page_config(page_title="딥러닝 품질 상태 예측", layout="wide")

# Sidebar 메뉴 설정
st.sidebar.title("메뉴")
menu = st.sidebar.radio(
    "항목",
    ("학습데이터 업로드", "예측데이터 업로드")
)

# 데이터 전처리 및 준비 함수
def prepare_data(csv_file):
    all_data = pd.read_csv(csv_file, encoding='cp949')
    numeric_features = all_data.select_dtypes(include=[np.number]).dropna()
    X = numeric_features.drop(columns=['품질상태']).values  # 입력 데이터
    y = numeric_features['품질상태'].values  # 출력 데이터

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, numeric_features.columns.drop('품질상태'), all_data

# 학습 결과 시각화 함수
def plot_history(history):
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
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

# 성능 지표 출력 함수
def display_metrics(model, X_test, y_test):
    # 모델 평가
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.write(f'**Test Loss:** {loss:.4f}, **Test Accuracy:** {accuracy:.4f}')

    # 예측 및 성능 지표 계산
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

# 메뉴 1: TXT 파일들을 업로드하여 품질 상태를 예측
if menu == "학습데이터 업로드":
    st.title("학습데이터 업로드")
    uploaded_file = st.file_uploader("(학습용)", type=["csv"], key="first_file")

    if uploaded_file is not None:
        X_train, X_test, y_train, y_test, scaler, numeric_features_columns, original_data = prepare_data(uploaded_file)

        # 학습 데이터를 session_state에 저장
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.scaler = scaler
        st.session_state.numeric_features_columns = numeric_features_columns
        st.session_state.original_data = original_data

        # 모델 구성 함수
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
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(32)
            test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0.001,
                patience=10,
                restore_best_weights=True
            )

            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            )

            history_obj = model.fit(
                train_dataset,
                epochs=100,
                callbacks=[early_stopping, reduce_lr],
                validation_data=test_dataset,
                verbose=0
            )

            # history를 딕셔너리 형태로 저장
            history = {
                'accuracy': history_obj.history['accuracy'],
                'val_accuracy': history_obj.history['val_accuracy'],
                'loss': history_obj.history['loss'],
                'val_loss': history_obj.history['val_loss']
            }

        # 학습 결과 시각화 및 session_state에 저장
        plot_history(history)

        # 성능 지표 출력
        display_metrics(model, X_test, y_test)

        # 모델과 히스토리 session_state에 저장
        st.session_state.model = model
        st.session_state.history = history
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

        # 학습 완료 후 자동으로 예측 페이지로 이동
        st.success("학습완료. 예측데이터 업로드 칸으로 이동하시게 용사여....")

    # 학습된 히스토리가 있을 경우, 페이지 다시 로드 시에도 시각화 및 성능 지표 출력
    elif "history" in st.session_state:
        st.write("이전 학습 결과를 시각화합니다.")
        plot_history(st.session_state.history)

        # 이전 성능 지표 출력
        display_metrics(st.session_state.model, st.session_state.X_test, st.session_state.y_test)

# 메뉴 2: 새로운 CSV 파일 업로드 (예측용)
if menu == "예측데이터 업로드":
    st.title("예측데이터 업로드")
    new_file = st.file_uploader("(예측용)", type=["csv"], key="new_file")

    if new_file is not None:
        new_data = pd.read_csv(new_file, encoding='cp949')
        numeric_features_new = new_data.select_dtypes(include=[np.number]).dropna()
        numeric_features_new = numeric_features_new[st.session_state.numeric_features_columns]
        X_new = numeric_features_new.values
        X_new_scaled = st.session_state.scaler.transform(X_new)

        # 모델을 session_state에서 불러오기
        model = st.session_state.model

        # 예측 수행
        y_new_pred_prob = model.predict(X_new_scaled)
        reconstruction_error_new = np.mean(np.abs(X_new_scaled - model.predict(X_new_scaled)), axis=1)
        threshold = np.mean(reconstruction_error_new) + 2 * np.std(reconstruction_error_new)
        new_anomalies = np.where(reconstruction_error_new > threshold)[0]

        st.write("**새로운 데이터의 이상치 인덱스 (첫 번째 열 값):**")
        anomaly_indices = new_data.iloc[new_anomalies].iloc[:, 0].values
        st.write(np.unique(anomaly_indices))
