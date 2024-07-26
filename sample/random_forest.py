import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

# 데이터 파일을 제대로 불러오기 위해 파일 경로와 인코딩을 확인합니다.
df = pd.read_csv("C:\\Users\\freeman\\Desktop\\빅브라더\\sample\\data_mv_sv_dv_ut_lt_pca_no_NTC.csv", encoding='cp949')

# 데이터의 구조를 확인하기 위해 처음 몇 줄을 출력합니다.
print(df.head())

# '품질상태' 열이 실제로 존재하는지 확인합니다.
if '품질상태' not in df.columns:
    raise KeyError("Column '품질상태' does not exist in the dataset.")

# 무의미한 변수 제거
df = df.drop(['파일명'], axis=1)

# 타겟 변수를 매핑합니다.
mapping_dict = {'0': 0, '1': 1}
df['품질상태'] = df['품질상태'].apply(lambda x: mapping_dict[str(x)])

# 특성 열과 타겟 열을 설정합니다.
feature_columns = list(df.columns.difference(['품질상태']))
X = df[feature_columns]
y = df['품질상태']

# 스케일링
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 특성 선택: 랜덤 포레스트를 사용하여 중요한 특성 선택
selector = RandomForestClassifier(n_estimators=100, random_state=42)
selector.fit(X, y)
sfm = SelectFromModel(selector, threshold='median')
X_transformed = sfm.transform(X)

# 학습 데이터와 평가 데이터로 분할합니다.
train_x, test_x, train_y, test_y = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)  # 데이터 개수 확인

# 랜덤 포레스트 분류기를 생성하고 학습합니다.
clf = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=42)
clf.fit(train_x, train_y)

# 평가 데이터로 예측을 수행하고 정확도를 출력합니다.
predict2 = clf.predict(test_x)
accuracy = accuracy_score(test_y, predict2)
print("Accuracy: ", accuracy)

# 로스 값을 계산하고 출력합니다.
predict_proba = clf.predict_proba(test_x)
loss = log_loss(test_y, predict_proba)
print("Log Loss: ", loss)
