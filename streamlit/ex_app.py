import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import joblib

# txtToDFPipline.py의 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import txtDatasToDFPipline as pipLineOne

def prepare_deepLearning_data(dataFrame):
    all_data = dataFrame

    # 특징 선택 (불필요한 열 제거 등)
    selected_features = all_data.drop(columns=['품질상태'])  # 품질상태를 제외한 특징 선택

    # 숫자 데이터만 선택
    numeric_features = selected_features.select_dtypes(include=[np.number])

    # 딥러닝의 입력 데이터와 정답 데이터 생성
    X = numeric_features.values  # 입력 데이터
    y = all_data['품질상태'].values  # 출력 데이터

    # 테스트 데이터와 트레이닝 데이터로 분할
    X_train, X_test_full, Y_train, Y_test_full = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test, X_val, Y_test, Y_val = train_test_split(X_test_full, Y_test_full, test_size=0.5, random_state=42)

    # 데이터 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled= scaler.transform(X_val)

    return X_train_scaled, X_test_scaled, X_val_scaled, Y_train, Y_test, Y_val, scaler, selected_features.columns

# 임시 디렉토리 생성 함수
def create_temp_dir():
    temp_dir = os.path.join(os.getcwd(), "temp_files")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    return temp_dir

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
        layers.Dense(8, activation='relu', input_shape=(X_train.shape[1],)),  # 첫 번째 은닉층
        layers.Dense(8, activation='relu'),  # 두 번째 은닉층
        layers.Dense(8, activation='relu'),  # 세 번째 은닉층
        layers.Dense(8, activation='relu'),   # 네 번째 은닉층
        layers.Dense(8, activation='relu'),   # 다섯 번째 은닉층
        layers.Dense(1, activation='sigmoid')  # 출력층 (이진 분류)
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

    if uploaded_files:
        temp_dir = create_temp_dir()  # 임시 디렉토리 생성
        data_frames = []
        data_frames = pipLineOne.makePreprocessedDf(txtFileList=uploaded_files)

        if data_frames is not None:
            # PCA 적용 및 학습 준비
            num_pca_components = 17
            pca_df_target, pca_df_fileName, pca_df_datas = pipLineOne.pca.distributeDataFrame(dataFrame=data_frames)

            pca_scalar_model = pipLineOne.pca.performStandScalar(df_datas=pca_df_datas)
            df_scaled = pca_scalar_model.transform(pca_df_datas)
            df_scaled = pd.DataFrame(df_scaled, columns=pca_df_datas.columns)

            be_scalar_model_save = input("StandardScalar 모델을 저장하시겠습니까? ")
            be_pcaModel_save = input("pca 모델을 저장하시겠습니까? ")
            
            if be_scalar_model_save == 'y':
                scalar_model_save_path = os.getcwd() + "\\MLP&ML\\Skl_models\\Scalar"
                scalar_model_name = "scalar_model"
                pipLineOne.pca.save_model(pca_scalar_model, scalar_model_save_path, scalar_model_name)

            pca_num_components = 7
    
            pca_model = pipLineOne.pca.make_pca_model(data_scaled = df_scaled, num_components = pca_num_components)
            
            if be_pcaModel_save == 'y':
                pca_model_save_path = os.getcwd() + "\\MLP&ML\\Skl_models\\Pca"
                pca_model_name = f"pca_model_{pca_num_components}" 
                pipLineOne.pca.save_model(pca_model, pca_model_save_path, pca_model_name)
            
            df_pca = pipLineOne.pca.make_pca_dataFrame(data_scaled=df_scaled, data_target=pca_df_target, data_fileName=pca_df_fileName, num_components=pca_num_components, pca_model= pca_model)

            # 학습용 데이터와 테스트 데이터를 분리
            X_train, X_test, X_val , y_train, y_test, Y_val, scaler, feature_columns = prepare_deepLearning_data(df_pca)

            # 특징과 레이블을 TensorFlow Dataset으로 변환합니다.
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(32)  # 배치 크기 조정
            test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)  # 배치 크기 조정
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(32)

            model = build_model(X_train.shape[1])

            # 모델 학습
            # Early Stopping 설정
            early_stopping = callbacks.EarlyStopping(
                min_delta=0.001,  # 최소한의 변화
                patience=50,      # 몇 번 연속으로 개선이 없는지
                restore_best_weights=True  # 최상의 가중치로 복원
            )
            reduce_lr = callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)

            with st.spinner('모델을 학습 중입니다...'):
                history = model.fit(train_dataset, epochs=100, callbacks=[early_stopping, reduce_lr], validation_data=val_dataset, verbose=0)
                #history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping, reduce_lr], verbose=0)

            # 학습 결과 및 성능 지표 표시
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
        
        # 데이터 전처리
        dataFrame2 = prepare_data(txt_file_path, new_file.name)

        if dataFrame2 is not None:
            pca_df_target, pca_df_fileName, pca_df_datas = pipLineOne.pca.distributeDataFrame(dataFrame=dataFrame2)
            pca_scalar_model = joblib.load("C:\\Users\\freeman\\Desktop\\빅브라더\\MLP&ML\\Skl_models\\Scalar\\scalar_model.pkl")
            pca_df_scaled = pca_scalar_model.transform(pca_df_datas)

            # 예측 수행
            model = st.session_state.model
            y_new_pred_prob = model.predict(pca_df_scaled)
            y_new_pred = (y_new_pred_prob > 0.5).astype(int).flatten()

            st.write(f"**예측 결과:** {y_new_pred}")
