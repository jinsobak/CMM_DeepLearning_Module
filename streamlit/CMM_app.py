import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import joblib

# txtToDFPipline.py의 경로 추가
sys.path.append('C:\\Users\\freeman\\Desktop\\빅브라더\\MLP&ML\\codes_박진서')
sys.path.append('C:\\Users\\freeman\\Desktop\\빅브라더\\MLP&ML\\EDA')

from txtDatasToDFPipline import extract_data_from_file, DFtoModifiedDF, pca


# 임시 디렉토리 생성 함수
def create_temp_dir():
    temp_dir = os.path.join(os.getcwd(), "temp_files")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    return temp_dir

# 데이터 전처리 및 준비 함수
def prepare_data(txt_file_path, txt_file_name):
    try:
        # 1차 전처리
        dataFrame1 = extract_data_from_file(file_path=txt_file_path, fileName=txt_file_name)

        # 품번 검사
        if dataFrame1['품번'][0] != '45926-4G100':
            st.error(f"{txt_file_name}의 품번이 맞지 않습니다.")
            return None

        # 2차 전처리
        labels = ['----|', '---|', '--|', '-|', '>|<', '|+', '|++', '|+++', '|++++']
        dataFrame2 = DFtoModifiedDF(dataFrame=dataFrame1, fileName=txt_file_name, labels=labels)
        return dataFrame2

    except Exception as e:
        st.error(f"파일을 처리하는 중 오류가 발생했습니다: {e}")
        return None

# 학습 결과 시각화 함수
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs_range, acc, label='Training Accuracy')
        ax.plot(epochs_range, val_acc, label='Validation Accuracy')
        ax.legend(loc='lower right')
        ax.set_title('Training and Validation Accuracy')
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs_range, loss, label='Training Loss')
        ax.plot(epochs_range, val_loss, label='Validation Loss')
        ax.legend(loc='upper right')
        ax.set_title('Training and Validation Loss')
        st.pyplot(fig)

# 성능 지표 출력 함수
def display_metrics(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.write(f'**Test Loss:** {loss:.4f}, **Test Accuracy:** {accuracy:.4f}')

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    cm = confusion_matrix(y_test, y_pred)

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Confusion Matrix:**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

    with col2:
        true_negatives = cm[0, 0]
        false_positives = cm[0, 1]
        false_negatives = cm[1, 0]
        true_positives = cm[1, 1]
        total_data = true_negatives + false_positives + false_negatives + true_positives

        st.write(f"**정확도(Accuracy):** {accuracy*100:.2f}%")
        st.write(f"정상을 불량으로 오판한 경우: {false_positives}회")
        st.write(f"불량을 정상으로 오판한 경우: {false_negatives}회")
        st.write(f"정상 데이터를 올바르게 예측한 경우: {true_negatives}회")
        st.write(f"불량 데이터를 올바르게 예측한 경우: {true_positives}회")

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        st.metric("Accuracy", f"{acc*100:.2f}%")
        st.metric("Precision", f"{precision*100:.2f}%")
        st.metric("Recall", f"{recall*100:.2f}%")
        st.metric("F1 Score", f"{f1*100:.2f}%")

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
    uploaded_files = st.file_uploader(
        "여러 txt 파일을 선택하세요", type=["txt"], accept_multiple_files=True)

    if uploaded_files is not None:
        temp_dir = create_temp_dir()  # 임시 디렉토리 생성
        data_frames = []
        for uploaded_file in uploaded_files:
            txt_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(txt_file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            dataFrame2 = prepare_data(txt_file_path, uploaded_file.name)
            if dataFrame2 is not None:
                data_frames.append(dataFrame2)

        if data_frames:
            full_data = pd.concat(data_frames, ignore_index=True)
            st.write("모든 파일이 성공적으로 결합되었습니다.")
            
            num_pca_components = 7
            pca_df_target, pca_df_fileName, pca_df_datas = pca.distributeDataFrame(dataFrame=full_data)
            pca_scalar_model = joblib.load("C:\\Users\\freeman\\Desktop\\빅브라더\\MLP&ML\\Skl_models\\Scalar\\scalar_model.pkl")
            pca_df_scaled = pca_scalar_model.transform(pca_df_datas)
            
            pca_model = joblib.load(f"C:\\Users\\freeman\\Desktop\\빅브라더\\MLP&ML\\Skl_models\\Pca\\pca_model_{num_pca_components}.pkl")
            pca_dataFrame = pca.make_pca_dataFrame(data_scaled=pca_df_scaled, data_target=pca_df_target, data_fileName=pca_df_fileName, num_components=num_pca_components, pca_model=pca_model)

            X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(pca_dataFrame.values[:, :-1], pca_dataFrame.values[:, -1], test_size=0.2, random_state=42)

            model = build_model(X_train.shape[1])

            early_stopping = callbacks.EarlyStopping(min_delta=0.001, patience=50, restore_best_weights=True)
            reduce_lr = callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)

            with st.spinner('모델을 학습 중입니다...'):
                history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping, reduce_lr], verbose=0)

            plot_history(history)
            display_metrics(model, X_test, y_test)
            st.session_state.model = model
            st.session_state.history = history
            st.success("학습 완료.")

# 메뉴 2: 예측 데이터 업로드
if menu == "예측데이터 업로드":
    st.title("예측데이터 업로드")
    new_file = st.file_uploader(
        "예측할 txt 파일을 업로드하세요", type=["txt"])

    if new_file is not None:
        temp_dir = create_temp_dir()  # 임시 디렉토리 생성
        txt_file_path = os.path.join(temp_dir, new_file.name)
        with open(txt_file_path, 'wb') as f:
            f.write(new_file.getbuffer())
        
        dataFrame2 = prepare_data(txt_file_path, new_file.name)

        if dataFrame2 is not None:
            pca_df_target, pca_df_fileName, pca_df_datas = pca.distributeDataFrame(dataFrame=dataFrame2)
            pca_scalar_model = joblib.load("C:\\Users\\freeman\\Desktop\\빅브라더\\MLP&ML\\Skl_models\\Scalar\\scalar_model.pkl")
            pca_df_scaled = pca_scalar_model.transform(pca_df_datas)

            model = st.session_state.model
            y_new_pred_prob = model.predict(pca_df_scaled)
            y_new_pred = (y_new_pred_prob > 0.5).astype(int).flatten()

            st.write(f"**예측 결과:** {y_new_pred}")
