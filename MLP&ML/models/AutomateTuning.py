import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import optuna

# 데이터 전처리 및 준비 함수


def prepare_data(csv_file):
    all_data = pd.read_csv(csv_file, encoding='cp949')
    selected_features = all_data.drop(columns=['품질상태'])
    numeric_features = selected_features.select_dtypes(include=[np.number])
    X = numeric_features.values
    y = all_data['품질상태'].values
    X_train, X_test_full, Y_train, Y_test_full = train_test_split(
        X, y, test_size=0.2, random_state=42)
    X_test, X_val, Y_test, Y_val = train_test_split(
        X_test_full, Y_test_full, test_size=0.5, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_test_scaled, X_val_scaled, Y_train, Y_test, Y_val, scaler, selected_features.columns

# Optuna의 objective 함수 정의


def objective(trial):
    # 하이퍼파라미터 샘플링
    num_layers = trial.suggest_int('num_layers', 1, 5)  # 은닉층 수
    units = trial.suggest_int('units', 4, 64, log=True)  # 유닛 수
    learning_rate = trial.suggest_float(
        'learning_rate', 1e-5, 1e-1, log=True)  # 학습률
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)  # 드롭아웃 비율
    batch_size = trial.suggest_categorical(
        'batch_size', [16, 32, 64, 128])  # 배치 사이즈
    epochs = trial.suggest_int('epochs', 10, 100)  # 에포크 수

    # 데이터 준비
    X_train, X_test, X_val, y_train, y_test, Y_val, scaler, feature_columns = prepare_data(
        'C:\\Users\\ddc4k\\OneDrive\\Desktop\\빅브라더\\MLP&ML\\datas\\data_jd_hd_delete_material_no_NTC_pca_component_7.csv')

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)).shuffle(len(X_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (X_val, Y_val)).batch(batch_size)

    # 모델 구성
    model = models.Sequential()
    model.add(layers.Dense(units, activation='relu',
              input_shape=(X_train.shape[1],)))
    model.add(layers.Dropout(dropout_rate))  # 드롭아웃 적용

    for _ in range(num_layers - 1):
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1, activation='sigmoid'))

    # 모델 컴파일
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # EarlyStopping 설정
    early_stopping = EarlyStopping(
        min_delta=0.001, patience=10, restore_best_weights=True)

    # 모델 학습
    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset,
              callbacks=[early_stopping], verbose=0)

    # 검증 데이터로 성능 평가
    val_loss, val_accuracy = model.evaluate(val_dataset, verbose=0)

    return val_loss

# 메인 함수


def main():
    # Optuna 스터디 생성
    study = optuna.create_study(direction='minimize')  # 손실을 최소화하는 방향으로 탐색
    study.optimize(objective, n_trials=500)  # 300번의 시도 진행

    print("Best trial:")
    trial = study.best_trial
    print(f"  Loss: {trial.value}")
    print("  Best hyperparameters: ", trial.params)

    # 최적의 하이퍼파라미터로 모델 재학습
    X_train, X_test, X_val, y_train, y_test, Y_val, scaler, feature_columns = prepare_data(
        'C:\\Users\\ddc4k\\OneDrive\\Desktop\\빅브라더\\MLP&ML\\datas\\data_jd_hd_delete_material_no_NTC_pca_component_7.csv')

    best_units = trial.params['units']
    best_layers = trial.params['num_layers']
    best_learning_rate = trial.params['learning_rate']
    best_dropout_rate = trial.params['dropout_rate']
    best_batch_size = trial.params['batch_size']
    best_epochs = trial.params['epochs']

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)).shuffle(len(X_train)).batch(best_batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (X_test, y_test)).batch(best_batch_size)

    model = models.Sequential()
    model.add(layers.Dense(best_units, activation='relu',
              input_shape=(X_train.shape[1],)))
    model.add(layers.Dropout(best_dropout_rate))

    for _ in range(best_layers - 1):
        model.add(layers.Dense(best_units, activation='relu'))
        model.add(layers.Dropout(best_dropout_rate))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 모델 학습
    early_stopping = EarlyStopping(
        min_delta=0.001, patience=10, restore_best_weights=True)
    model.fit(train_dataset, epochs=best_epochs,
              validation_data=test_dataset, callbacks=[early_stopping])

    # 모델 평가
    loss, accuracy = model.evaluate(test_dataset)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

    # 예측 결과
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # 성능 지표
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f'Accuracy: {acc}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')


if __name__ == "__main__":
    main()