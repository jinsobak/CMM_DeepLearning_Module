import pandas as pd
import os
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as pre
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 데이터 전처리 및 준비
def prepare_data(csv_files):
    # CSV 파일을 읽어들여 DataFrame으로 변환
    all_data = pd.read_csv(csv_files, encoding='cp949')
    all_data = pd.DataFrame(all_data)

    # 딥러닝의 입력 데이터와 정답 데이터 생성
    X = all_data.drop(columns=['품질상태', '품명'])  # 입력 데이터
    y = all_data['품질상태']  # 출력 데이터

    # 테스트 데이터와 트레이닝 데이터로 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scalar = pre.StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)

    return X_train, X_test, y_train, y_test

class testClassifier:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.classifier = None

    def fit(self, train_data, train_label, num_epochs, batch_size = 1):
        self.classifier.fit(train_data, train_label, epochs=num_epochs, batch_size=batch_size)

    def predict(self, test_data):
        predictions = self.classifier.predict(test_data)
        return predictions

    # 모델 구축
    def build_model(self):
        input_layer = tf.keras.layers.Input(shape=[self.input_dim,])

        activation_func_relu = tf.keras.activations.relu
        hidden_layer1 = tf.keras.layers.Dense(units = 64, activation=activation_func_relu)(input_layer)
        hidden_layer2 = tf.keras.layers.Dense(units = 32, activation=activation_func_relu)(hidden_layer1)
        hidden_layer3 = tf.keras.layers.Dense(units = 16, activation=activation_func_relu)(hidden_layer2)

        activation_func_sig = tf.keras.activations.sigmoid
        output_layer = tf.keras.layers.Dense(units=1, activation=activation_func_sig)(hidden_layer3)
        classifier_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

        print(classifier_model.summary())

        optimize_alg = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        classifier_model.compile(optimizer=optimize_alg, loss=loss_func, metrics=['accuracy'])

        self.classifier = classifier_model

        # classifier = Sequential([
        #     Dense(64, activation='relu', input_dim=input_dim),
        #     Dense(32, activation='relu'),
        #     Dense(1, activation='sigmoid')  # 시그모이드 함수를 사용하여 확률값을 출력
        # ])
        # classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # self.classifier = classifier

if __name__ == "__main__":
    # 데이터 파일 경로
    csv_files = os.getcwd() + '\\test\\test_ld.csv'

    # 데이터 전처리 및 준비
    X_train, X_test, y_train, y_test = prepare_data(csv_files)


    # 모델 구축
    input_dim = X_train.shape[1]
    print(input_dim)
    classifier = testClassifier(input_dim, 1)
    classifier.build_model()

    # 모델 학습
    #model.fit(X_train, y_train, epochs=100, batch_size=1)
    classifier.fit(X_train, y_train, num_epochs=20, batch_size=1)

    # 모델 평가
    test_loss, test_accuracy = classifier.classifier.evaluate(X_test, y_test)
    print("Test Accuracy:", test_accuracy)

    prd = classifier.classifier.predict(X_test)
    zipped = zip(y_test, prd)
    for y, predict in zipped:
        if predict > 0.5:
            print("%.3f => 1 y => %d" %(predict, y))
        else:
            print("%.3f => 0 y => %d" %(predict, y))