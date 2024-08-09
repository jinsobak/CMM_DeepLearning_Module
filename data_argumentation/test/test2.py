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
