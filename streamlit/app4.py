import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# 데이터 전처리 및 준비 함수
def prepare_data(file, file_type='csv'):
    try:
        if file_type == 'csv':
            all_data = pd.read_csv(file, encoding='cp949')
        else:
            st.error("지원하지 않는 파일 형식입니다.")
            return None
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return None

    # 숫자형 데이터 선택 및 결측치 제거
    numeric_features = all_data.select_dtypes(include=[np.number]).dropna()

    if '품질상태' not in numeric_features.columns:
        st.error("'품질상태' 컬럼이 존재하지 않습니다.")
        return None

    # 입력 데이터와 출력 데이터 분리
    X = numeric_features.drop(columns=['품질상태']).values  # 입력 데이터
    y = numeric_features['품질상태'].values  # 출력 데이터

    # 학습용 데이터와 테스트용 데이터 분리
    X_train, X_test_full, y_train, y_test_full = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test_full, y_test_full, test_size=0.5, random_state=42)

    # 데이터 정규화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    return X_train_scaled, X_test_scaled, X_val_scaled, y_train, y_test, y_val, scaler, numeric_features.columns.drop('품질상태'), all_data


# 학습 결과 시각화 함수
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

# 성능 지표 출력 함수
def display_metrics(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.write(f'**Test Loss:** {loss:.4f}, **Test Accuracy:** {accuracy:.4f}')

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    cm = confusion_matrix(y_test, y_pred)
    st.write("**Confusion Matrix:**")
    st.write(cm)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    st.write(f'**Accuracy:** {acc:.4f}')
    st.write(f'**Precision:** {precision:.4f}')
    st.write(f'**Recall:** {recall:.4f}')
    st.write(f'**F1 Score:** {f1:.4f}')

# 모델 구성 함수
def build_model(input_shape):
    model = models.Sequential([
        layers.Dense(8, activation='relu', input_shape=(input_shape,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Streamlit 앱 설정
st.set_page_config(page_title="딥러닝 품질 상태 예측", layout="wide")

# Sidebar 메뉴 설정
st.sidebar.title("메뉴")
menu = st.sidebar.radio("항목", ("학습데이터 업로드", "예측데이터 업로드"))

# 메뉴 1: 학습 데이터 업로드
if menu == "학습데이터 업로드":
    st.title("학습데이터 업로드")
    uploaded_file = st.file_uploader("학습용 데이터 업로드 (CSV 파일)", type=["csv"])

    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1]
        data = prepare_data(uploaded_file, file_type=file_type)

        if data:
            X_train, X_test, X_val, y_train, y_test, y_val, scaler, numeric_features_columns, original_data = data
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.X_val = X_val
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.y_val = y_val
            st.session_state.scaler = scaler
            st.session_state.numeric_features_columns = numeric_features_columns

            model = build_model(X_train.shape[1])

            early_stopping = callbacks.EarlyStopping(min_delta=0.001, patience=50, restore_best_weights=True)
            reduce_lr = callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)

            with st.spinner('모델을 학습 중입니다...'):
                history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
                                    callbacks=[early_stopping, reduce_lr], verbose=0)

            plot_history(history)
            display_metrics(model, X_test, y_test)
            st.session_state.model = model
            st.session_state.history = history
            st.success("학습 완료. 예측데이터 업로드 페이지로 이동하세요.")

# 메뉴 2: 예측 데이터 업로드
if menu == "예측데이터 업로드":
    st.title("예측데이터 업로드")
    new_file = st.file_uploader("예측용 데이터 업로드 (CSV 파일)", type=["csv"])

    if new_file is not None:
        file_type = new_file.name.split('.')[-1]
        data = prepare_data(new_file, file_type=file_type)

        if data:
            _, _, _, _, _, _, scaler, numeric_features_columns, original_data = data
            try:
                numeric_features_new = original_data.select_dtypes(include=[np.number]).dropna()
                numeric_features_new = numeric_features_new[numeric_features_columns]
                X_new = numeric_features_new.values
                X_new_scaled = scaler.transform(X_new)

                model = st.session_state.model
                y_new_pred_prob = model.predict(X_new_scaled)
                y_new_pred = (y_new_pred_prob > 0.5).astype(int).flatten()

                cm = confusion_matrix(original_data['품질상태'], y_new_pred)
                st.write("**Confusion Matrix:**")
                st.write(cm)

                acc = accuracy_score(original_data['품질상태'], y_new_pred)
                precision = precision_score(original_data['품질상태'], y_new_pred, zero_division=0)
                recall = recall_score(original_data['품질상태'], y_new_pred, zero_division=0)
                f1 = f1_score(original_data['품질상태'], y_new_pred, zero_division=0)

                st.write(f'**Accuracy:** {acc:.4f}')
                st.write(f'**Precision:** {precision:.4f}')
                st.write(f'**Recall:** {recall:.4f}')
                st.write(f'**F1 Score:** {f1:.4f}')

            except Exception as e:
                st.error(f"예측 중 오류가 발생했습니다: {e}")
