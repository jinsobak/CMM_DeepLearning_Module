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
import optuna
import tempfile

# txtToDFPipline.py의 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import txtDatasToDFPipline as txtFilesPipLine
import txtToDFPipline as txtOnepipLine
import PCA_visualization as pca

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
def plot_history(history, early_stopping):
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
        ax.axvline(early_stopping.stopped_epoch - early_stopping.patience, 0.0, 1.0, linestyle='--', linewidth=1, label='best Epoch')
        ax.legend(loc='lower right')
        ax.set_title('Training and Validation Accuracy')
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs_range, loss, label='Training Loss')
        ax.plot(epochs_range, val_loss, label='Validation Loss')
        ax.axvline(early_stopping.stopped_epoch - early_stopping.patience, 0.0, 1.0, linestyle='--', linewidth=1, label='best Epoch')
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

# Optuna의 objective 함수 정의
def objective(trial, X_train, X_val, y_train, Y_val):
    # 하이퍼파라미터 샘플링
    num_layers = trial.suggest_int('num_layers', 2, 6)  # 은닉층 수
    units = trial.suggest_int('units', 8, 128, log=True)  # 유닛 수
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-2, log=True)  # 학습률
    #dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)  # 드롭아웃 비율
    batch_size = trial.suggest_categorical('batch_size', [16, 32])  # 배치 사이즈
    epochs = 100

    # 모델 구성
    model = models.Sequential()
    model.add(layers.Dense(units, activation='relu', input_shape=(X_train.shape[1],)))
    #model.add(layers.Dropout(dropout_rate))
    
    for _ in range(num_layers - 1):
        model.add(layers.Dense(units, activation='relu'))
        #model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss='binary_crossentropy', metrics=['accuracy'])

    # EarlyStopping 설정
    early_stopping = callbacks.EarlyStopping(
        min_delta=0.001, patience=10, restore_best_weights=True)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(batch_size)

    # 모델 학습
    history = model.fit(train_dataset, epochs=epochs, callbacks=[early_stopping], verbose=0)

    val_loss, val_accuracy = model.evaluate(val_dataset, verbose=0)
    return val_accuracy

def SaveSessionStates():
    if 'piplineStep' not in st.session_state:
        st.session_state.piplineStep = 0
    if "history" not in st.session_state:
        st.session_state.history = None
    if "DLModelSaveStep" not in st.session_state:
        st.session_state.DLModelSaveStep = 0

def learning_button_click():
    print("learningButtonClicked")
    st.session_state.piplineStep = 1

def SavePredictSessionStates():
    if 'predictStep' not in st.session_state:
        st.session_state.predictStep = 0
        
def Predict_button_click():
    print("predictButtonClicked")
    st.session_state.predictStep = 1

# Streamlit 앱 설정
st.set_page_config(page_title="딥러닝 품질 상태 예측", layout="wide")

# Sidebar 메뉴 설정
st.sidebar.title("메뉴")
menu = st.sidebar.radio("항목", ("학습데이터 업로드", "예측데이터 업로드"))

SaveSessionStates()

# 메뉴 1: 학습 데이터 업로드
if menu == "학습데이터 업로드":
    
    st.title("학습데이터 업로드")
    st.session_state.uploaded_files = st.file_uploader(
        "여러 txt 파일을 선택하세요", type=["txt"], accept_multiple_files=True)

    uploadedFiles = st.session_state.uploaded_files

    make_standardScalar_model_button = st.checkbox("정규화 모델을 저장하시겠습니까?")
    make_pca_model_button = st.checkbox("PCA 모델을 저장하시겠습니까?")
    hyperParamOptimizeNum = st.number_input("하이퍼파라미터 최적화 횟수를 입력하세요", min_value=1)
    
    do_learning_button = st.button("학습 시작", on_click=learning_button_click)

    if uploadedFiles:
        temp_dir = create_temp_dir()  # 임시 디렉토리 생성

        if st.session_state.piplineStep == 1:
            with st.spinner("데이터를 변환하는 중입니다..."):
                data_frames = txtFilesPipLine.makePreprocessedDf(txtFileList=uploadedFiles)

                if data_frames is not None:
                    # PCA 적용 및 학습 준비
                    st.session_state.pca_num_components = 7
                    pca_df_target, pca_df_fileName, pca_df_datas = pca.distributeDataFrame(dataFrame=data_frames)

                    scalar_model = pca.Make_StandScalar_model(df_datas=pca_df_datas)
                    if make_standardScalar_model_button:
                        scalar_model_save_path = os.getcwd() + "\\MLP&ML\\Skl_models\\Scalar"
                        scalar_model_name = "scalar_model"
                        pca.save_model(scalar_model, scalar_model_save_path, scalar_model_name)
                                
                    df_scaled = scalar_model.transform(pca_df_datas)
                    df_scaled = pd.DataFrame(df_scaled, columns=pca_df_datas.columns)

                    pca_model = pca.Make_pca_model(data_scaled = df_scaled, num_components = st.session_state.pca_num_components) 
                    if make_pca_model_button:   
                        pca_model_save_path = os.getcwd() + "\\MLP&ML\\Skl_models\\Pca"
                        pca_model_name = f"pca_model" 
                        pca.save_model(pca_model, pca_model_save_path, pca_model_name)
                    
                    st.session_state.df_pca = pca.make_pca_dataFrame(data_scaled=df_scaled, data_target=pca_df_target, 
                                                    data_fileName=pca_df_fileName,
                                                    pca_model=pca_model)

                    st.session_state.piplineStep = 2

                    st.write("데이터 변환이 완료되었습니다.\n")   
           
        if st.session_state.piplineStep == 2:
            X_train, X_test, X_val , y_train, y_test, Y_val, scaler, feature_columns = prepare_deepLearning_data(st.session_state.df_pca)

            st.session_state.X_test = X_test
            st.session_state.Y_test = y_test
            
            # Optuna를 사용한 하이퍼파라미터 튜닝
            with st.spinner('최적의 하이퍼파라미터를 탐색하는 중입니다...'):
                study = optuna.create_study(direction='maximize')
                study.optimize(lambda trial: objective(trial, X_train, X_val, y_train, Y_val), n_trials=hyperParamOptimizeNum)

            best_trial = study.best_trial
            st.write(f"최적의 하이퍼파라미터: {best_trial.params}")

            # 최적의 하이퍼파라미터를 이용해 모델 생성 및 학습
            best_units = best_trial.params['units']
            best_layers = best_trial.params['num_layers']
            best_learning_rate = best_trial.params['learning_rate']
            #best_dropout_rate = best_trial.params['dropout_rate']
            best_batch_size = best_trial.params['batch_size']

            model = models.Sequential()
            model.add(layers.Dense(best_units, activation='relu', input_shape=(X_train.shape[1],)))
            #model.add(layers.Dropout(best_dropout_rate))

            for _ in range(best_layers - 1):
                model.add(layers.Dense(best_units, activation='relu'))
                #model.add(layers.Dropout(best_dropout_rate))

            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_learning_rate), 
                        loss='binary_crossentropy', metrics=['accuracy'])

            st.session_state.early_stopping = callbacks.EarlyStopping(monitor='val_loss' ,min_delta=0.001, patience=10, restore_best_weights=True)

            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(best_batch_size)
            test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(best_batch_size)
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(best_batch_size)

            with st.spinner('최적의 하이퍼파라미터로 학습 중입니다...'):
                st.session_state.history = model.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=[st.session_state.early_stopping], verbose=0)

            st.session_state.model = model
            st.session_state.piplineStep = 3

        if st.session_state.piplineStep == 3:
            # 학습 결과 및 성능 지표 표시
            plot_history(st.session_state.history, st.session_state.early_stopping)
            display_metrics(st.session_state.model, st.session_state.X_test, st.session_state.Y_test)

            st.write("학습 완료.")

            #모델 저장 여부 결정
            model_save_button = st.button("학습된 모델을 저장하시겠습니까?")
            if model_save_button:
                modelName = "MLPModel.keras"
                st.session_state.model.save(os.getcwd() + "\\DeeplearningModels\\" + modelName)
                st.success("모델 저장 완료")


# 메뉴 2: 예측 데이터 업로드
if menu == "예측데이터 업로드":
    st.title("예측데이터 업로드")

    SavePredictSessionStates()
    
    new_file = st.file_uploader(
        "예측할 txt 파일을 업로드하세요", type=["txt"], accept_multiple_files=True)
    
    do_predict_button = st.button("예측 시작", on_click=Predict_button_click)
    
    if new_file is not None:
        #print(st.session_state.predictStep)
        temp_dir = create_temp_dir()  # 임시 디렉토리 생성
        
        if st.session_state.predictStep == 1:
            # 데이터 전처리
            dataFrame_predict = txtFilesPipLine.makePreprocessedDfForPredict(new_file)
            dataFrame_number = dataFrame_predict['번호']
            st.dataframe(dataFrame_number)
            dataFrame_predict = dataFrame_predict.drop(columns=['번호'])
            if dataFrame_predict is not None:
                pca_df_target, pca_df_fileName, pca_df_datas = pca.distributeDataFrame(dataFrame=dataFrame_predict)

                #저장되어 있는 정규화 모델을 통해 정규화 실시
                pca_scalar_model = joblib.load(os.getcwd() +"\\MLP&ML\\Skl_models\\Scalar\\scalar_model.pkl")
                pca_df_scaled = pca_scalar_model.transform(pca_df_datas)

                #저장되어 있는 PCA모델을 통해 정규화된 예측용 데이터프레임에 PCA기법 수행
                pca_model = joblib.load(os.getcwd() + f"\\MLP&ML\\Skl_models\\Pca\\pca_model.pkl")
                pca_dataFrame = pca.make_pca_dataFrame(data_scaled = pca_df_scaled, data_target = pca_df_target, 
                                                    data_fileName = pca_df_fileName, pca_model = pca_model)

                df_predict = pca_dataFrame.drop(columns=['품질상태'])

                # 예측 수행
                model = models.load_model(os.getcwd() + "\\DeepLearningModels\\MLPModel.keras")

                y_new_pred_prob = model.predict(df_predict)

                zipped = zip(pca_df_fileName, y_new_pred_prob)
                y_new_pred = (y_new_pred_prob > 0.5).astype(int).flatten()
                
                df_result = pd.DataFrame()
                for fileName, predict in zipped:
                    verdict = lambda x : "**정상**" if x >= 0.5 else "**불량**"
                    df_verdict = pd.DataFrame([[fileName, predict, verdict(predict)]], columns = ["파일명", "예측확률", "예측결과"])
                    print(df_verdict)
                    df_result = pd.concat([df_result, df_verdict], ignore_index = False)
                    #st.write(f"부품이 정상일 확률: %.3f 판정: %s" %(predict, verdict(predict)))
                df_result.reset_index(inplace=True)
                df_result.drop(columns=['index'], inplace=True)
                df_result = pd.concat([df_result, dataFrame_number], axis=1)
                
                st.session_state.df_result = df_result
                st.session_state.predictStep = 2

        if st.session_state.predictStep == 2:
            df_result = st.session_state.df_result
            
            col1, col2 = st.columns(2)
            
            with col1:
                predicted_datas = st.dataframe(df_result, 
                            column_config={
                                "파일명": st.column_config.Column(
                                    width="large"
                                )
                            }, 
                            selection_mode="single-row",
                            on_select="rerun")
            
            selected_row = predicted_datas.selection.rows
            
            with col2:
                if len(selected_row) > 0:
                    file = new_file[df_result.iloc[selected_row[0]]['번호']]
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
                        temp_file.write(file.read())
                        temp_file_path = temp_file.name  # 임시 파일 경로
                    
                    df_arrangedTxtData = txtOnepipLine.extract_data_from_file(file_path = temp_file_path, fileName=file.name)
                    
                    st.write(f"파일명: {df_result.iloc[selected_row].loc[:, '파일명'].to_string().split(' ')[4]}")
                    st.write(f"품명: {df_arrangedTxtData['품명'][0]}")
                    st.write(f"측정 시간: {df_arrangedTxtData['측정시간'][0]}")
                    st.write(f"측정자: {df_arrangedTxtData['측정자'][0]}")
                    st.dataframe(df_arrangedTxtData.loc[:, '번호':])
                