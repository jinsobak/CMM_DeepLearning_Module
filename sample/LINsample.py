import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# 데이터 전처리 및 준비
def prepare_data(csv_file):
    # CSV 파일을 읽어들여 DataFrame으로 변환
    all_data = pd.read_csv(csv_file, encoding='cp949')

    # 딥러닝의 입력 데이터와 정답 데이터 생성
    X = all_data.drop(columns=['품질상태', '품명']).values  # 입력 데이터
    y = all_data['품질상태'].values  # 출력 데이터

    # 테스트 데이터와 트레이닝 데이터로 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 데이터 스케일링
    scalar = StandardScaler()
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scalar

# CSV 파일 경로
csv_file = "C:\\Users\\freeman\\Desktop\\빅브라더\\sample\\data_dv_hd.csv"

# 데이터 전처리
X_train, X_test, y_train, y_test, scalar = prepare_data(csv_file)

# 선형 회귀 모델 구성
model = LinearRegression()

# 모델 학습
model.fit(X_train, y_train)

# 모델 평가
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f'Training Score: {train_score}')
print(f'Test Score: {test_score}')

# 불량 판별
# 새로운 데이터를 모델에 입력하여 불량 여부를 판별합니다.
new_data = [
    # 새로운 데이터셋을 리스트로 저장
]

if len(new_data) == 0:
    print("새로운 데이터가 없습니다.")
else:
    new_data_scaled = scalar.transform(new_data)  # 새로운 데이터 전처리
    predictions = model.predict(new_data_scaled)
    for i, pred in enumerate(predictions):
        if pred >= 0.5:
            print(f'데이터 {i + 1}은(는) 불량입니다.')
        else:
            print(f'데이터 {i + 1}은(는) 정상입니다.')
