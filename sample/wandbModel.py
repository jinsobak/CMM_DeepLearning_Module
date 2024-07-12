import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import wandb
from wandb.integration.keras import WandbMetricsLogger
import os

# WandB 로그인 및 프로젝트 설정
wandb.login()
wandb.require("core")

# 데이터 전처리 및 준비 함수
def prepare_data(csv_file):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File not found: {csv_file}")

    # CSV 파일을 읽어들여 DataFrame으로 변환
    all_data = pd.read_csv(csv_file, encoding='cp949')

    # 특징 선택 (불필요한 열 제거 등)
    selected_features = all_data.drop(columns=['품질상태'])  # 품질상태를 제외한 특징 선택

    # 숫자 데이터만 선택
    numeric_features = selected_features.select_dtypes(include=[np.number])

    # 딥러닝의 입력 데이터와 정답 데이터 생성
    X = numeric_features.values  # 입력 데이터
    y = all_data['품질상태'].values  # 출력 데이터

    # 테스트 데이터와 트레이닝 데이터로 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 데이터 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# CSV 파일 경로
csv_file = "C:\\Users\\freeman\\Desktop\\빅브라더\\sample\\data_mv_sv_dv_ut_lt_hd_no_NTC.csv"

# 데이터 전처리
X_train, X_test, y_train, y_test, scaler = prepare_data(csv_file)

# 특징과 레이블을 TensorFlow Dataset으로 변환합니다.
train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)).shuffle(len(X_train)).batch(32)  # 배치 크기 조정
test_dataset = tf.data.Dataset.from_tensor_slices(
    (X_test, y_test)).batch(32)  # 배치 크기 조정

# 하이퍼파라미터 정의
sweep_config = {
    'method': 'random',  # random search
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'   
    },
    'parameters': {
        'dense_layer_1': {
            'values': [256, 512, 1024]
        },
        'dense_layer_2': {
            'values': [128, 256, 512]
        },
        'dense_layer_3': {
            'values': [64, 128, 256]
        },
        'dense_layer_4': {
            'values': [32, 64, 128]
        },
        'dropout_rate': {
            'values': [0.3, 0.4, 0.5]
        },
        'learning_rate': {
            'values': [0.001, 0.0005, 0.0001]
        },
        'epochs': {
            'values': [50, 100, 200]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="my_project")

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        # 딥러닝 모델 구성
        model = models.Sequential([
            Input(shape=(X_train.shape[1],)),
            layers.Dense(config.dense_layer_1, activation='relu'),
            layers.Dropout(config.dropout_rate),
            layers.Dense(config.dense_layer_2, activation='relu'),
            layers.Dropout(config.dropout_rate),
            layers.Dense(config.dense_layer_3, activation='relu'),
            layers.Dropout(config.dropout_rate),
            layers.Dense(config.dense_layer_4, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        # 모델 컴파일
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # Early Stopping 설정
        early_stopping = EarlyStopping(
            min_delta=0.001,
            patience=50,
            restore_best_weights=True
        )

        # 모델 학습
        model.fit(train_dataset,
                  epochs=config.epochs,
                  validation_data=test_dataset,
                  callbacks=[early_stopping, WandbMetricsLogger()])

        # 모델 평가
        loss, accuracy = model.evaluate(test_dataset)
        print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Sweep 실행
wandb.agent(sweep_id, function=train, count=20)
