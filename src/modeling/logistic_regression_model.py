import joblib
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.logistic_regression import LogisticRegression as HandmadeModel

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(ROOT, 'data', 'processed')
MODELS_PATH = os.path.join(ROOT, 'models')

# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
data_frame = pd.read_csv(os.path.join(DATA_PATH, 'processed.csv'))

x = data_frame.drop(columns=['Class'])
y = data_frame['Class']

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

# Scaling for Logistic Regression
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# using Logistic Regression (Fraud = 1, Non-Fraud = 0)
model = LogisticRegression(class_weight='balanced')
model.fit(x_train_scaled, y_train)
handmade = HandmadeModel()
handmade.fit(x_train_scaled, y_train)

y_predict = model.predict(x_test_scaled)
y_predict_handmade = handmade.predict(x_test_scaled)

accuracy = accuracy_score(y_test, y_predict) * 100
accuracy_handmade = accuracy_score(y_test, y_predict_handmade) * 100
print(f"Accuracy: {accuracy:.2f}")
print(f"Accuracy from Handmade Model: {accuracy_handmade:.2f}")

joblib.dump(model, os.path.join(MODELS_PATH, 'logistic_regression.pkl'))
joblib.dump(handmade, os.path.join(MODELS_PATH, 'handmade_logistic_regression.pkl'))
