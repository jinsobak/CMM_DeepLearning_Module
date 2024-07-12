import pandas as pd

# 파일 경로
ng_file = r"C:\git_folder\CMM_DeepLearning_Module\MLP\datas\data_mv_sv_dv_ut_lt_hd_only_NG.csv"
ok_file = r"C:\git_folder\CMM_DeepLearning_Module\MLP\datas\data_mv_sv_dv_ut_lt_hd_only_OK.csv"
ntc_file = r"C:\git_folder\CMM_DeepLearning_Module\MLP\datas\data_mv_sv_dv_ut_lt_hd_only_NTC.csv"

# 데이터 불러오기
ng_data = pd.read_csv(ng_file, encoding='ISO-8859-1')
ok_data = pd.read_csv(ok_file, encoding='ISO-8859-1')
ntc_data = pd.read_csv(ntc_file, encoding='ISO-8859-1')

# 데이터 확인
print(ng_data.head())
print(ok_data.head())
print(ntc_data.head())

# 라벨 추가
ng_data['label'] = 'NG'
ok_data['label'] = 'OK'
ntc_data['label'] = 'NTC'

# 데이터 병합
combined_data = pd.concat([ng_data, ok_data, ntc_data], ignore_index=True)

# 데이터 비율 맞추기
min_len = min(len(ng_data), len(ok_data), len(ntc_data))

ng_sample = ng_data.sample(min_len)
ok_sample = ok_data.sample(min_len)
ntc_sample = ntc_data.sample(min_len)

balanced_data = pd.concat([ng_sample, ok_sample, ntc_sample], ignore_index=True)

# 특성과 라벨 분리
X = balanced_data.drop('label', axis=1)
y = balanced_data['label']

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_evaluate(X, y, labels):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for labels {labels}: {accuracy:.2f}")

# NG vs OK
ng_ok_data = balanced_data[balanced_data['label'].isin(['NG', 'OK'])]
X_ng_ok = ng_ok_data.drop('label', axis=1)
y_ng_ok = ng_ok_data['label']
train_and_evaluate(X_ng_ok, y_ng_ok, ['NG', 'OK'])

# NG vs NTC
ng_ntc_data = balanced_data[balanced_data['label'].isin(['NG', 'NTC'])]
X_ng_ntc = ng_ntc_data.drop('label', axis=1)
y_ng_ntc = ng_ntc_data['label']
train_and_evaluate(X_ng_ntc, y_ng_ntc, ['NG', 'NTC'])

# OK vs NTC
ok_ntc_data = balanced_data[balanced_data['label'].isin(['OK', 'NTC'])]
X_ok_ntc = ok_ntc_data.drop('label', axis=1)
y_ok_ntc = ok_ntc_data['label']
train_and_evaluate(X_ok_ntc, y_ok_ntc, ['OK', 'NTC'])

from sklearn.ensemble import VotingClassifier

# 개별 모델
ng_ok_model = RandomForestClassifier(random_state=42)
ng_ntc_model = RandomForestClassifier(random_state=42)
ok_ntc_model = RandomForestClassifier(random_state=42)

# 학습
ng_ok_model.fit(X_ng_ok, y_ng_ok)
ng_ntc_model.fit(X_ng_ntc, y_ng_ntc)
ok_ntc_model.fit(X_ok_ntc, y_ok_ntc)

# 앙상블 모델
ensemble_model = VotingClassifier(estimators=[
    ('ng_ok', ng_ok_model),
    ('ng_ntc', ng_ntc_model),
    ('ok_ntc', ok_ntc_model)
], voting='hard')

# 최종 데이터로 학습 및 평가
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
ensemble_model.fit(X_train, y_train)
y_pred = ensemble_model.predict(X_test)

# 최종 정확도
final_accuracy = accuracy_score(y_test, y_pred)
print(f"Final Ensemble Accuracy: {final_accuracy:.2f}")
