import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

# 데이터 파일을 제대로 불러오기 위해 파일 경로와 인코딩을 확인합니다.
df = pd.read_csv(
    "C:\\Users\\ddc4k\\OneDrive\\Desktop\\빅브라더\\MLP&ML\\datas\\data_jd_hd_delete_material_no_NTC_pca_component_17.csv", encoding='cp949')


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

# 학습 데이터와 평가 데이터로 분할합니다.
train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)  # 데이터 개수 확인

# KNN 분류기를 생성하고 학습합니다.
knn = KNeighborsClassifier(n_neighbors=5)  # K는 5로 설정, 필요에 따라 조정 가능
knn.fit(train_x, train_y)

# 평가 데이터로 예측을 수행하고 정확도를 출력합니다.
predict2 = knn.predict(test_x)
accuracy = accuracy_score(test_y, predict2)

print("Accuracy: ", accuracy)

# 로스 값을 계산하고 출력합니다.
predict_proba = knn.predict_proba(test_x)
loss = log_loss(test_y, predict_proba)
print("Log Loss: ", loss)

# 추가 성능 지표 계산 및 출력
precision = precision_score(test_y, predict2)
recall = recall_score(test_y, predict2)
f1 = f1_score(test_y, predict2)
roc_auc = roc_auc_score(test_y, predict_proba[:, 1])

print("Precision: ", precision)
print("Recall: ", recall)
print("F1-Score: ", f1)
print("ROC-AUC: ", roc_auc)

# 컨퓨전 메트릭스 계산
conf_matrix = confusion_matrix(test_y, predict2)
print(conf_matrix)
