import joblib
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

scaler = joblib.load('./scaler.bin')

X_test = pd.read_csv('test/test_features.csv', sep=';')
Y_test = pd.read_csv('test/test_labels.csv', sep=';')

model = joblib.load("./model.pkl")

y_pred = model.predict(X_test)

y_true_rounded = np.round(Y_test['Температура_A3_°C'].to_numpy())
y_pred_rounded = np.round(y_pred)

acc = accuracy_score(y_true_rounded, y_pred_rounded)
print(f"Точность до градуса: {acc:.2f}")

