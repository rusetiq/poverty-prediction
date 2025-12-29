import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

df = pd.read_csv(url, names=columns, na_values=' ?', skipinitialspace=True)

df = df.dropna()

df['poverty_risk'] = (df['income'] == '<=50K').astype(int)

categorical_columns = ['workclass', 'education', 'marital_status', 'occupation', 
                       'relationship', 'race', 'sex', 'native_country']

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

feature_columns = ['age', 'workclass', 'education_num', 'marital_status', 'occupation',
                   'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
                   'hours_per_week']

X = df[feature_columns]
y = df['poverty_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

print("=" * 60)
print("POVERTY RISK PREDICTION MODEL RESULTS")
print("Using UCI Adult Census Income Dataset")
print("=" * 60)
print(f"\nDataset Size: {len(df)} records")
print(f"Training Set: {len(X_train)} records")
print(f"Test Set: {len(X_test)} records")
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not At Risk (>50K)', 'At Risk (<=50K)']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nFeature Importance (Coefficients):")
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)
print(feature_importance.to_string(index=False))

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS FROM TEST SET")
print("=" * 60)
sample_indices = np.random.choice(len(X_test), 5, replace=False)
for idx in sample_indices:
    actual = y_test.iloc[idx]
    predicted = y_pred[idx]
    probability = y_pred_proba[idx]
    print(f"Actual: {'At Risk' if actual == 1 else 'Not At Risk'}, "
          f"Predicted: {'At Risk' if predicted == 1 else 'Not At Risk'}, "
          f"Risk Probability: {probability:.2%}")
