import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# 데이터 불러오기
file_path = '/mnt/data/data_jd_hd_delete_material_no_NTC_pca_component_17.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# 피처와 라벨 분리
X = data.iloc[:, 1:-1]  # component_0 ~ component_16
y = data.iloc[:, -1]    # 품질상태

# 1. 데이터 전처리 (결측값 처리)
# 결측값이 있는 경우, 중간값으로 대체
X.fillna(X.median(), inplace=True)

# 2. 데이터 스케일링 (StandardScaler 사용)
scaler = StandardScaler()

# 3. 특징 선택 (SelectKBest 사용하여 상위 k개의 피처 선택)
selector = SelectKBest(score_func=f_classif, k='all')  # 모든 피처 사용, 필요시 k값을 변경

# 4. 로지스틱 회귀 모델 생성
model = LogisticRegression(max_iter=1000)

# 5. 파이프라인 설정
pipeline = Pipeline([
    ('scaler', scaler),
    ('selector', selector),
    ('model', model)
])

# 6. 하이퍼파라미터 튜닝 (GridSearchCV 사용)
param_grid = {
    'model__C': [0.01, 0.1, 1, 10, 100],  # 규제 강도
    'model__penalty': ['l1', 'l2', 'none'],  # 규제 방식 (L1, L2, 규제 없음)
    'model__solver': ['liblinear', 'saga']  # 최적화 알고리즘
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y)

# 7. 최적 하이퍼파라미터와 성능 출력
print("Best parameters found:", grid_search.best_params_)
print(f"Best cross-validation accuracy: {grid_search.best_score_:.2f}")

# 8. 최적 모델로 예측 수행
best_model = grid_search.best_estimator_

# 학습 데이터와 테스트 데이터 분할 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 훈련
best_model.fit(X_train, y_train)

# 예측
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]  # 클래스 1에 대한 확률

# 9. 성능 평가
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'ROC AUC Score: {roc_auc:.2f}')

# 분류 리포트 출력
print(classification_report(y_test, y_pred))

# 혼동 행렬 출력
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# 10. 교차 검증 결과 출력
cv_scores = cross_val_score(best_model, X, y, cv=10, scoring='accuracy')
print(f'Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%')
