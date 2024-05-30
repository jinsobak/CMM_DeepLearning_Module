import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
import wandb
from wandb.integration.keras import WandbMetricsLogger

# wandb 초기화
wandb.init(project='Ensemble Example(Train)')
wandb.run.name = 'ensemble wandb'
wandb.run.save()


def prepare_data(csv_file):
    # CSV 파일을 읽어들여 DataFrame으로 변환
    all_data = pd.read_csv(csv_file, encoding='cp949')

    # 특징 선택 (불필요한 열 제거 등)
    selected_features = all_data.drop(
        columns=['품질상태'])  # 품질상태와 품명을 제외한 특징 선택

    # 딥러닝의 입력 데이터와 정답 데이터 생성
    X = selected_features.values  # 입력 데이터
    y = all_data['품질상태'].values  # 출력 데이터

    # 클래스별로 데이터를 분리
    normal_idx = np.where(y == 1)
    abnormal_idx = np.where(y == 0)
    unknown_idx = np.where(y == 2)

    X_normal = X[normal_idx]
    y_normal = y[normal_idx]
    X_abnormal = X[abnormal_idx]
    y_abnormal = y[abnormal_idx]
    X_unknown = X[unknown_idx]
    y_unknown = y[unknown_idx]

    # 각각 테스트 데이터와 트레이닝 데이터로 분할
    X_normal_train, X_normal_test, y_normal_train, y_normal_test = train_test_split(
        X_normal, y_normal, test_size=0.2, random_state=42)
    X_abnormal_train, X_abnormal_test, y_abnormal_train, y_abnormal_test = train_test_split(
        X_abnormal, y_abnormal, test_size=0.2, random_state=42)
    X_unknown_train, X_unknown_test, y_unknown_train, y_unknown_test = train_test_split(
        X_unknown, y_unknown, test_size=0.2, random_state=42)

    # 데이터 스케일링
    scalar = StandardScaler()
    X_normal_train_scaled = scalar.fit_transform(X_normal_train)
    X_normal_test_scaled = scalar.transform(X_normal_test)
    X_abnormal_train_scaled = scalar.fit_transform(X_abnormal_train)
    X_abnormal_test_scaled = scalar.transform(X_abnormal_test)
    X_unknown_train_scaled = scalar.fit_transform(X_unknown_train)
    X_unknown_test_scaled = scalar.transform(X_unknown_test)

    return (X_normal_train_scaled, X_normal_test_scaled, y_normal_train, y_normal_test), \
           (X_abnormal_train_scaled, X_abnormal_test_scaled, y_abnormal_train, y_abnormal_test), \
           (X_unknown_train_scaled, X_unknown_test_scaled,
            y_unknown_train, y_unknown_test), scalar


def build_and_train_model(X_train, y_train, X_test, y_test, class_weights):
    # Early Stopping 설정
    early_stopping = EarlyStopping(
        min_delta=0.001,  # 최소한의 변화
        patience=20,      # 몇 번 연속으로 개선이 없는지
        restore_best_weights=True  # 최상의 가중치로 복원
    )

    # 딥러닝 모델 구성
    model = models.Sequential([
        layers.Dense(64, activation='relu',
                     input_shape=(X_train.shape[1],)),  # 입력층
        layers.Dense(32, activation='relu'),  # 은닉층
        layers.Dense(16, activation='relu'),  # 은닉층 추가
        layers.Dense(1, activation='sigmoid')  # 출력층 (이진 분류)
    ])

    # 모델 컴파일
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall'])

    # 모델 학습
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weights,
        callbacks=[early_stopping, WandbMetricsLogger()]
    )

    # 모델 평가
    loss, accuracy, precision, recall = model.evaluate(X_test)
    print(
        f'Test Loss: {loss}, Test Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}')

    # wandb 설정 업데이트 및 로그 기록
    wandb.config.update({
        "epochs": 100,
        "batch_size": 32
    })

    wandb.log({
        "Test Loss": loss,
        "Test Accuracy": accuracy,
        "Test Precision": precision,
        "Test Recall": recall
    })

    return model


# CSV 파일 경로
csv_file = "C:\\Users\\ddc4k\\Onedrive\\Desktop\\빅브라더\\sample\\data_dv_hd_with_NTC.csv"

# 데이터 전처리
(normal_train, normal_test, abnormal_train, abnormal_test,
 unknown_train, unknown_test, scalar) = prepare_data(csv_file)

# 클래스 가중치 계산
normal_class_weights = compute_class_weight(
    'balanced', classes=np.unique(normal_train[1]), y=normal_train[1])
abnormal_class_weights = compute_class_weight(
    'balanced', classes=np.unique(abnormal_train[1]), y=abnormal_train[1])
unknown_class_weights = compute_class_weight(
    'balanced', classes=np.unique(unknown_train[1]), y=unknown_train[1])

normal_class_weights = dict(enumerate(normal_class_weights))
abnormal_class_weights = dict(enumerate(abnormal_class_weights))
unknown_class_weights = dict(enumerate(unknown_class_weights))

# 모델 학습
normal_model = build_and_train_model(
    *normal_train, *normal_test, normal_class_weights)
abnormal_model = build_and_train_model(
    *abnormal_train, *abnormal_test, abnormal_class_weights)
unknown_model = build_and_train_model(
    *unknown_train, *unknown_test, unknown_class_weights)

# 예측 수행
normal_preds = normal_model.predict(normal_test[0])
abnormal_preds = abnormal_model.predict(abnormal_test[0])
unknown_preds = unknown_model.predict(unknown_test[0])

# 예측 결과 결합 (앙상블)
final_predictions = np.argmax(
    np.array([normal_preds, abnormal_preds, unknown_preds]), axis=0)

# 예측 결과를 실제 라벨로 변환
final_labels = np.where(final_predictions == 0, 1,
                        np.where(final_predictions == 1, 0, 2))

# 정확도 평가
accuracy = accuracy_score(normal_test[1], final_labels)
precision = precision_score(normal_test[1], final_labels, average='weighted')
recall = recall_score(normal_test[1], final_labels, average='weighted')
print(
    f'Ensemble Model Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}')
