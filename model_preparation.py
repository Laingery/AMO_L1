import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib


X_train = pd.read_csv('train/train_features.csv', sep=';')
Y_train = pd.read_csv('train/train_labels.csv', sep=';')

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, Y_train['Температура_A3_°C'].to_numpy())


joblib.dump(model, './model.pkl')
