import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

def prepare_data(ng_file, ok_file, ntc_file):
    ng_data = pd.read_csv(ng_file, encoding='cp949')
    ok_data = pd.read_csv(ok_file, encoding='cp949')
    ntc_data = pd.read_csv(ntc_file, encoding='cp949')

    ng_data['label'] = 0
    ok_data['label'] = 1
    ntc_data['label'] = 2

    return ng_data, ok_data, ntc_data

def preprocess_data(ng_data, ok_data, ntc_data, labels):
    data = pd.concat([ng_data[ng_data['label'].isin(labels)],
                      ok_data[ok_data['label'].isin(labels)],
                      ntc_data[ntc_data['label'].isin(labels)]])
    
    selected_features = data.drop(columns=['label'])
    numeric_features = selected_features.select_dtypes(include=[np.number])

    X = numeric_features.values
    y = data['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def build_and_train_model(X_train, y_train, X_test, y_test):
    early_stopping = EarlyStopping(min_delta=0.001, patience=20, restore_best_weights=True)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, epochs=100, callbacks=[early_stopping], validation_data=test_dataset)

    loss, accuracy = model.evaluate(test_dataset)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

    return model

# 파일 경로
ng_file = r"C:\git_folder\CMM_DeepLearning_Module\MLP\datas\data_mv_sv_dv_ut_lt_hd_only_NG.csv"
ok_file = r"C:\git_folder\CMM_DeepLearning_Module\MLP\datas\data_mv_sv_dv_ut_lt_hd_only_OK.csv"
ntc_file = r"C:\git_folder\CMM_DeepLearning_Module\MLP\datas\data_mv_sv_dv_ut_lt_hd_only_NTC.csv"

# 데이터 전처리
ng_data, ok_data, ntc_data = prepare_data(ng_file, ok_file, ntc_file)

# NG vs OK
X_train_ng_ok, X_test_ng_ok, y_train_ng_ok, y_test_ng_ok, scaler = preprocess_data(ng_data, ok_data, ntc_data, [0, 1])
model_ng_ok = build_and_train_model(X_train_ng_ok, y_train_ng_ok, X_test_ng_ok, y_test_ng_ok)

# OK vs NTC
X_train_ok_ntc, X_test_ok_ntc, y_train_ok_ntc, y_test_ok_ntc, _ = preprocess_data(ng_data, ok_data, ntc_data, [1, 2])
model_ok_ntc = build_and_train_model(X_train_ok_ntc, y_train_ok_ntc, X_test_ok_ntc, y_test_ok_ntc)

# NG vs NTC
X_train_ng_ntc, X_test_ng_ntc, y_train_ng_ntc, y_test_ng_ntc, _ = preprocess_data(ng_data, ok_data, ntc_data, [0, 2])
model_ng_ntc = build_and_train_model(X_train_ng_ntc, y_train_ng_ntc, X_test_ng_ntc, y_test_ng_ntc)

# NTC 데이터를 예측
ntc_scaled = scaler.transform(ntc_data.drop(columns=['label']).select_dtypes(include=[np.number]).values)
y_pred_ntc_ng_ok = (model_ng_ok.predict(ntc_scaled) > 0.5).astype(int)
y_pred_ntc_ok_ntc = (model_ok_ntc.predict(ntc_scaled) > 0.5).astype(int)
y_pred_ntc_ng_ntc = (model_ng_ntc.predict(ntc_scaled) > 0.5).astype(int)

# Pseudo-Label 부여 (다수결)
pseudo_labels_ntc = np.round((y_pred_ntc_ng_ok + y_pred_ntc_ok_ntc + y_pred_ntc_ng_ntc) / 3).astype(int)

# NTC 데이터에 Pseudo-Label 추가
ntc_data['label'] = pseudo_labels_ntc

# NG와 OK 데이터에 Pseudo-Label이 부여된 NTC 데이터를 추가
pseudo_labeled_data = pd.concat([ng_data, ok_data, ntc_data])

# 재훈련을 위해 데이터 전처리
X_train_pseudo, X_test_pseudo, y_train_pseudo, y_test_pseudo, _ = preprocess_data(pseudo_labeled_data, ok_data, ntc_data, [0, 1])

# 모델 재훈련
model_pseudo = build_and_train_model(X_train_pseudo, y_train_pseudo, X_test_pseudo, y_test_pseudo)

# 최종 정확도 평가
y_pred_pseudo = (model_pseudo.predict(X_test_pseudo) > 0.5).astype(int)
final_accuracy = accuracy_score(y_test_pseudo, y_pred_pseudo)
print(f'Final Ensemble Accuracy: {final_accuracy}')
